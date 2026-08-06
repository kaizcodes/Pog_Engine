from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import requests
import shutil
import subprocess
import sys
import threading
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from pipeline_config import MODEL, JUDGE_MODEL, OLLAMA_URL

DEFAULT_CHUNK_MINUTES = 25
MIN_REPEAT_SENTENCE_WORDS = 3
MIN_SUBTITLE_WORDS = 3
MAX_SUBTITLE_WORDS = 6
MAX_SUBTITLE_LINE_CHARS = 42
# Words per "thought" for LLM discovery to read (see merge_entries_for_
# analysis()) - independent of the on-screen caption size above; the LLM
# does better with fuller thoughts than a stream of 3-6 word fragments.
TRANSCRIPT_MERGE_TARGET_WORDS = 30
# A gap this long between captions is a new-thought boundary when merging
# for LLM analysis, even before the target word count above is hit.
TRANSCRIPT_MERGE_MAX_GAP_MS = 2500
SCRIPT_DIR = Path(__file__).resolve().parent
ANALYZE_HIGHLIGHTS = SCRIPT_DIR / "analyze_highlights_emotion.py"
# Used only when count_audio_streams() below detects a single-track (Twitch-
# style) VOD - see make_extract_mic_bat_singletrack().
ISOLATE_VOCALS_SCRIPT = SCRIPT_DIR / "isolate_vocals.py"
GALLERY_DIR = Path(r"G:\pog_dev\gallery\best of")
GALLERY_IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}

# Edit these if your whisper.cpp install moves.
WHISPER_CLI = r"G:\pog_dev\models\Release\whisper-cli.exe"
WHISPER_MODEL = r"G:\pog_dev\models\ggml-large-v3.bin"
WHISPER_VAD = r"G:\pog_dev\models\ggml-silero-v6.2.0.bin"

# Noise gate before loudnorm: pushes down quiet background noise (keyboard,
# game bleed, room tone) instead of letting it get amplified into something
# the VAD mistakes for speech.
#   threshold: below this = "silence", attenuated. Raise (e.g. -30) if noise
#              leaks through; lower (e.g. -45) if quiet speech gets gated out.
#   ratio:     how hard the gate closes below threshold (higher = harder).
#   attack/release: ms for the gate to close/reopen.
NOISE_GATE_THRESHOLD_DB = -35
NOISE_GATE_RATIO = 8
NOISE_GATE_ATTACK_MS = 10
NOISE_GATE_RELEASE_MS = 200

@dataclass(frozen=True)
class SubtitleEntry:
    block: str
    start_ms: int
    end_ms: int

def parse_srt_time(value: str) -> int:
    """Convert HH:MM:SS,mmm into milliseconds."""
    match = re.fullmatch(r"\s*(\d+):(\d{2}):(\d{2}),(\d{1,3})\s*", value)
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")

    hours, minutes, seconds, milliseconds = match.groups()
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1_000
        + int(milliseconds.ljust(3, "0"))
    )

def format_srt_time(milliseconds: int) -> str:
    """Convert milliseconds into HH:MM:SS,mmm."""
    if milliseconds < 0:
        milliseconds = 0

    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

def format_plain_time(milliseconds: int) -> str:
    """Convert milliseconds into HH:MM:SS for transcript_part files."""
    hours, remainder = divmod(max(milliseconds, 0), 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds = remainder // 1_000
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def split_srt_blocks(content: str) -> list[str]:
    return [block.strip() for block in re.split(r"\r?\n\r?\n+", content.strip()) if block.strip()]

def read_text(path: Path) -> str:
    # utf-8-sig handles the BOM emitted by some Windows tools.
    return path.read_text(encoding="utf-8-sig")

def write_text_crlf(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    normalized = re.sub(r"\r?\n", "\r\n", content)
    path.write_text(normalized, encoding=encoding, newline="")

def clean_subtitle_lines(lines: Iterable[str]) -> list[str]:
    return [line.replace("â™ª", "").replace("♪", "") for line in lines]

def normalize_caption_text(lines: Iterable[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(clean_subtitle_lines(lines))).strip()

def normalize_repeated_sentence_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.casefold())).strip()

def is_repeat_sentence_candidate(text_key: str) -> bool:
    return len(text_key.split()) >= MIN_REPEAT_SENTENCE_WORDS

def remove_consecutive_repeated_sentences(text: str) -> tuple[str, int]:
    """Collapse only adjacent duplicate multi-word sentences inside one caption."""
    sentence_units = [match.group(0).strip() for match in re.finditer(r"[^.!?]+[.!?]*", text) if match.group(0).strip()]
    if len(sentence_units) <= 1:
        return text, 0

    kept_units: list[str] = []
    previous_key = ""
    removed_count = 0

    for sentence in sentence_units:
        sentence_key = normalize_repeated_sentence_key(sentence)
        if sentence_key and sentence_key == previous_key and is_repeat_sentence_candidate(sentence_key):
            removed_count += 1
            continue

        kept_units.append(sentence)
        previous_key = sentence_key

    return " ".join(kept_units), removed_count

NATURAL_BREAK_LEAD_WORDS = {
    "and", "but", "so", "because", "then", "or", "which", "that",
    "when", "if", "since", "while", "though", "although",
}

def find_natural_break_indices(words: list[str]) -> set[int]:
    """Word indices that make a reasonable pause point for a caption break:
    right after a comma, or right before a conjunction/subordinator that
    naturally starts a new breath group (e.g. "...the boss, {break} and
    then he just...")."""
    natural: set[int] = set()
    for index, word in enumerate(words):
        if word.rstrip("\"'").endswith(","):
            natural.add(index + 1)
        if index > 0 and word.strip(".,!?\"'").lower() in NATURAL_BREAK_LEAD_WORDS:
            natural.add(index)
    return natural

def split_sentence_into_natural_chunks(sentence: str) -> list[str]:
    """Splits one sentence into ~MIN_SUBTITLE_WORDS-MAX_SUBTITLE_WORDS-word
    pieces. Distributes words as evenly as possible across however many
    chunks the sentence needs, snapping each cut point to a nearby natural
    break (comma, conjunction) when one exists, rather than a hard
    mechanical every-N-words cut - the goal is captions that still read
    like natural phrases, not just short for the sake of short.
    """
    words = sentence.split()
    if len(words) <= MAX_SUBTITLE_WORDS:
        return [sentence] if words else []

    natural_breaks = find_natural_break_indices(words)
    n_chunks = math.ceil(len(words) / MAX_SUBTITLE_WORDS)
    ideal_size = len(words) / n_chunks

    cut_points: list[int] = []
    cursor = 0
    for chunk_num in range(1, n_chunks):
        ideal = round(chunk_num * ideal_size)
        # Keep every chunk (including what's left after this cut) within
        # [MIN, MAX] words where the remaining word count allows it.
        lo = cursor + MIN_SUBTITLE_WORDS
        hi = min(cursor + MAX_SUBTITLE_WORDS, len(words) - MIN_SUBTITLE_WORDS * (n_chunks - chunk_num))
        if hi < lo:
            hi = lo
        ideal = max(lo, min(ideal, hi))

        best_cut = ideal
        best_distance = None
        for candidate in natural_breaks:
            if lo <= candidate <= hi:
                distance = abs(candidate - ideal)
                if best_distance is None or distance < best_distance:
                    best_cut, best_distance = candidate, distance

        cut_points.append(best_cut)
        cursor = best_cut

    chunks: list[str] = []
    start = 0
    for cut in cut_points:
        chunks.append(" ".join(words[start:cut]))
        start = cut
    chunks.append(" ".join(words[start:]))
    return [chunk for chunk in chunks if chunk]

def split_caption_text(text: str, duration_ms: int) -> list[str]:
    """Split a long Whisper subtitle into short, natural-sounding captions
    (target MIN_SUBTITLE_WORDS-MAX_SUBTITLE_WORDS words each), without ever
    cutting in the middle of a sentence you're still speaking - sentence
    boundaries (.!?) are found first and never merged across; only text
    within one sentence gets subdivided further, and only when it's longer
    than MAX_SUBTITLE_WORDS words.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    raw_sentences = re.findall(r"[^.!?]+[.!?]+(?:['\"])?|[^.!?]+$", text)

    pieces: list[str] = []
    for sentence in raw_sentences or [text]:
        sentence = sentence.strip()
        if sentence:
            pieces.extend(split_sentence_into_natural_chunks(sentence))

    return [piece for piece in pieces if piece]

def wrap_caption_text(text: str) -> list[str]:
    wrapped = textwrap.wrap(
        text,
        width=MAX_SUBTITLE_LINE_CHARS,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped or [text]

def make_srt_block(index: int, start_ms: int, end_ms: int, text: str) -> str:
    return "\r\n".join(
        [
            str(index),
            f"{format_srt_time(start_ms)} --> {format_srt_time(end_ms)}",
            *wrap_caption_text(text),
        ]
    )

def split_caption_block(block: str, start_index: int) -> list[str]:
    lines = re.split(r"\r?\n", block)
    if len(lines) < 3 or "-->" not in lines[1]:
        lines[0] = str(start_index)
        return ["\r\n".join(lines)]

    start_text, end_text = [part.strip() for part in lines[1].split("-->", 1)]
    start_ms = parse_srt_time(start_text)
    end_ms = parse_srt_time(end_text)
    duration_ms = max(end_ms - start_ms, 1)
    pieces = split_caption_text(normalize_caption_text(lines[2:]), duration_ms)
    if not pieces:
        return []

    cursor_ms = start_ms
    split_blocks: list[str] = []

    for offset, piece in enumerate(pieces):
        if offset == len(pieces) - 1:
            piece_end_ms = end_ms
        else:
            piece_end_ms = start_ms + round(duration_ms * (offset + 1) / len(pieces))
            piece_end_ms = min(max(piece_end_ms, cursor_ms + 1), end_ms)

        split_blocks.append(make_srt_block(start_index + offset, cursor_ms, piece_end_ms, piece))
        cursor_ms = piece_end_ms

    return split_blocks

def fix_srt(input_path: Path) -> Path:
    """Fix bad Whisper SRT timestamps, collapse adjacent repeats, and renumber."""
    input_path = input_path.resolve()
    output_path = input_path.with_name(f"{input_path.stem}_fixed.srt")

    fixed_blocks: list[str] = []
    previous_end_ms = 0
    removed_sentence_count = 0

    for block in split_srt_blocks(read_text(input_path)):
        lines = re.split(r"\r?\n", block)
        if len(lines) < 2:
            continue

        index_line = lines[0]
        time_line = lines[1]
        if "-->" not in time_line:
            fixed_blocks.append(block)
            continue

        start_text, end_text = [part.strip() for part in time_line.split("-->", 1)]
        start_ms = parse_srt_time(start_text)
        end_ms = parse_srt_time(end_text)

        if start_ms == 0 and end_ms > 5 * 60_000:
            start_ms = previous_end_ms + 1
            end_ms = start_ms + 2_000
        elif start_ms >= end_ms:
            start_ms = previous_end_ms + 1
            end_ms = start_ms + 2_000
        elif start_ms < previous_end_ms:
            duration_ms = end_ms - start_ms
            if duration_ms > 5 * 60_000:
                duration_ms = 2_000
            start_ms = previous_end_ms + 1
            end_ms = start_ms + duration_ms

        previous_end_ms = end_ms
        subtitle_text = normalize_caption_text(lines[2:])
        subtitle_text, removed_in_block = remove_consecutive_repeated_sentences(subtitle_text)
        removed_sentence_count += removed_in_block
        fixed_blocks.append(
            "\r\n".join(
                [index_line, f"{format_srt_time(start_ms)} --> {format_srt_time(end_ms)}", subtitle_text]
            )
        )

    deduped_blocks: list[str] = []
    previous_text_key: str | None = None
    removed_block_count = 0

    for block in fixed_blocks:
        lines = re.split(r"\r?\n", block)
        text = normalize_caption_text(lines[2:]) if len(lines) > 2 else ""
        text_key = normalize_repeated_sentence_key(text)

        if text_key and text_key == previous_text_key and is_repeat_sentence_candidate(text_key):
            removed_block_count += 1
            continue

        deduped_blocks.append(block)
        previous_text_key = text_key or None

    renumbered_blocks: list[str] = []
    counter = 1
    for block in deduped_blocks:
        split_blocks = split_caption_block(block, counter)
        renumbered_blocks.extend(split_blocks)
        counter += len(split_blocks)

    if removed_block_count:
        print(f"Removed {removed_block_count} consecutive repeated subtitle block(s)")
    if removed_sentence_count:
        print(f"Removed {removed_sentence_count} consecutive repeated sentence(s)")
    split_count = len(renumbered_blocks) - len(deduped_blocks)
    if split_count:
        print(f"Split long subtitle text into {split_count} additional Resolve-friendly block(s)")

    write_text_crlf(output_path, "\r\n\r\n".join(renumbered_blocks), encoding="utf-8")
    print(f"Fixed file saved as: {output_path}")
    return output_path

def merge_entries_for_analysis(entries: list[SubtitleEntry]) -> list[tuple[int, str]]:
    """Regroups the short (3-6 word) on-screen captions back into fuller,
    more natural chunks for the LLM discovery passes to read - a stream of
    isolated 3-6 word fragments with no surrounding context is harder for a
    model to reason about than a couple of full sentences at a time. Each
    merged chunk keeps the timestamp of whichever caption started it, which
    stays a precise, real anchor for the anti-hallucination timestamp check
    downstream in analyze_highlights_emotion.py.

    A chunk closes (a new one starts) when adding the next caption would
    push it past TRANSCRIPT_MERGE_TARGET_WORDS words, when there's a long
    silence gap before the next caption (a natural pause = a natural new
    thought), or once a sentence has just ended and the chunk already has
    a reasonable amount of text - whichever comes first. Returns
    (start_ms, text) tuples.
    """
    merged: list[tuple[int, str]] = []
    buffer_words: list[str] = []
    buffer_start_ms = 0
    previous_end_ms: int | None = None

    for entry in entries:
        lines = re.split(r"\r?\n", entry.block)
        text = " ".join(lines[2:]).strip() if len(lines) >= 3 else ""
        if not text:
            continue

        gap_ms = entry.start_ms - previous_end_ms if previous_end_ms is not None else 0
        ends_sentence = bool(buffer_words) and buffer_words[-1].rstrip("\"'").endswith((".", "!", "?"))
        would_exceed_target = bool(buffer_words) and len(buffer_words) + len(text.split()) > TRANSCRIPT_MERGE_TARGET_WORDS

        should_close = buffer_words and (
            gap_ms > TRANSCRIPT_MERGE_MAX_GAP_MS
            or would_exceed_target
            or (ends_sentence and len(buffer_words) >= MIN_SUBTITLE_WORDS)
        )
        if should_close:
            merged.append((buffer_start_ms, " ".join(buffer_words)))
            buffer_words = []

        if not buffer_words:
            buffer_start_ms = entry.start_ms

        buffer_words.extend(text.split())
        previous_end_ms = entry.end_ms

    if buffer_words:
        merged.append((buffer_start_ms, " ".join(buffer_words)))

    return merged

def split_srt_into_chunks(input_path: Path, chunk_minutes: int = DEFAULT_CHUNK_MINUTES) -> list[Path]:
    """Create transcript_partN.txt files grouped into N-minute chunks."""
    entries: list[SubtitleEntry] = []

    for block in split_srt_blocks(read_text(input_path)):
        lines = re.split(r"\r?\n", block)
        if len(lines) < 2 or "-->" not in lines[1]:
            continue

        start_text, end_text = [part.strip() for part in lines[1].split("-->", 1)]
        entries.append(SubtitleEntry(block=block, start_ms=parse_srt_time(start_text), end_ms=parse_srt_time(end_text)))

    if not entries:
        print("No subtitle entries found.")
        return []

    video_start_ms = entries[0].start_ms
    video_end_ms = entries[-1].end_ms
    total_minutes = max(video_end_ms - video_start_ms, 0) / 60_000
    num_chunks = max(1, math.ceil(total_minutes / chunk_minutes))
    chunks: list[list[str]] = [[] for _ in range(num_chunks)]

    merged_entries = merge_entries_for_analysis(entries)

    for start_ms, text in merged_entries:
        output_line = f"[{format_plain_time(start_ms)}]\r\n{text}"

        offset_minutes = (start_ms - video_start_ms) / 60_000
        chunk_index = math.floor(offset_minutes / chunk_minutes)
        chunk_index = min(max(chunk_index, 0), num_chunks - 1)
        chunks[chunk_index].append(output_line)

    output_paths: list[Path] = []
    print("\nCreated:")
    for index, chunk in enumerate(chunks, start=1):
        part_path = input_path.resolve().parent / f"transcript_part{index}.txt"
        write_text_crlf(part_path, "\r\n\r\n".join(chunk), encoding="utf-8")
        output_paths.append(part_path)
        print(f"  {part_path.name}")

    print(f"\nChunk length: {chunk_minutes} minute(s) - {num_chunks} part(s) created.")
    print(
        f"Merged {len(entries)} on-screen caption(s) into {len(merged_entries)} "
        f"block(s) for analysis (~{TRANSCRIPT_MERGE_TARGET_WORDS} words each)."
    )
    print("Each block keeps the real start timestamp of the caption that began it.")
    return output_paths

def batch_quote(path: Path | str) -> str:
    return str(path).replace('"', '""')

def count_audio_streams(video_path: Path) -> int | None:
    """How many audio streams video_path has, via ffprobe. Returns None if
    ffprobe isn't available or the probe fails, so callers can fall back to
    the safer assumption (a locally recorded VOD with separate game/mic
    tracks) instead of silently guessing this is a single-track Twitch-style
    VOD - see organize_video()."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "json",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        return len(json.loads(result.stdout).get("streams", []))
    except (ValueError, AttributeError):
        return None

def make_extract_mic_bat_multitrack(target_folder: Path, base_name: str, video_suffix: str) -> str:
    """A locally recorded OBS VOD: desktop/game audio and mic are already on
    separate tracks, so this just pulls track index 1 (mic) straight out -
    unchanged from the original single-track-unaware version of this
    function, just parametrized on the real file extension instead of a
    hardcoded .mp4."""
    mic_wav_name = f"{base_name}_mic.wav"
    video_path = target_folder / f"{base_name}{video_suffix}"
    wav_path = target_folder / mic_wav_name
    audio_filters = (
        f"agate=threshold={NOISE_GATE_THRESHOLD_DB}dB:"
        f"ratio={NOISE_GATE_RATIO}:"
        f"attack={NOISE_GATE_ATTACK_MS}:"
        f"release={NOISE_GATE_RELEASE_MS},"
        #"loudnorm=I=-16:TP=-1.5:LRA=11"
    )
    return f'''@echo off
echo This VOD has separate game/mic audio tracks - extracting and gating/
echo normalizing the mic track (track index 1) from {base_name}{video_suffix} ...
ffmpeg -i "{batch_quote(video_path)}" -map 0:a:1 -ar 16000 -ac 1 -af "{audio_filters}" "{batch_quote(wav_path)}"
if errorlevel 1 (
    echo.
    echo ERROR: ffmpeg failed. Make sure ffmpeg is installed and on your PATH,
    echo and that the video is in this folder.
    if not "%RUN_ALL%"=="1" pause
    exit /b 1
)
echo.
echo Done! Mic audio saved as: {mic_wav_name}
echo You can now drag {mic_wav_name} onto 2_TranscribeAudio.bat
if not "%RUN_ALL%"=="1" pause
exit /b 0
'''

def make_extract_mic_bat_singletrack(target_folder: Path, base_name: str, video_suffix: str,
                                      isolate_script: Path) -> str:
    """A downloaded Twitch-style VOD: game audio, music, alerts, and mic are
    all flattened into one track, so there's no track to just pull out.
    Instead: extract the full mix at a Demucs-friendly rate (44.1kHz
    stereo - downsampling before separation would hurt separation quality)
    to Wave64 (.w64), then hand off to isolate_vocals.py, which runs the
    actual voice isolation and finishes the result into the same 16kHz
    mono + noise-gated format make_extract_mic_bat_multitrack() produces
    directly. Everything downstream only ever looks for *_mic.wav (see
    find_mic_wav() / find_run_all_file()'s "mic_wav" kind in
    analyze_highlights_emotion.py / this file), so it never needs to know
    which path produced it.

    .w64 not .wav here: standard RIFF/WAV caps data size at 4GB (32-bit
    size field), which a long Twitch VOD's stereo PCM mix blows past -
    ffmpeg then writes a structurally-broken file Demucs can't load.
    Wave64 uses a 64-bit size field, same signed 16-bit PCM payload Demucs
    (via soundfile/libsndfile) reads without any backend change."""
    mic_wav_name = f"{base_name}_mic.wav"
    mixed_wav_name = f"{base_name}_mixed_full.w64"
    video_path = target_folder / f"{base_name}{video_suffix}"
    mixed_wav_path = target_folder / mixed_wav_name
    mic_wav_path = target_folder / mic_wav_name
    return f'''@echo off
echo This VOD has one merged audio track (Twitch-style) - extracting the full
echo mix first, then isolating the streamer's voice from game audio, music,
echo and alerts. This takes longer than a normal extraction and, on the
echo very first run, downloads a small separation model (~80MB, needs
echo internet once).
echo.
echo Step 1/2: extracting full mix from {base_name}{video_suffix} ...
ffmpeg -y -i "{batch_quote(video_path)}" -map 0:a:0 -ar 44100 -ac 2 -f w64 "{batch_quote(mixed_wav_path)}"
if errorlevel 1 (
    echo.
    echo ERROR: ffmpeg failed to extract the mixed audio track. Make sure
    echo ffmpeg is installed and on your PATH, and that the video is in this
    echo folder.
    if not "%RUN_ALL%"=="1" pause
    exit /b 1
)
echo.
echo Step 2/2: isolating vocals (this can take a while on a long VOD)...
python "{batch_quote(isolate_script)}" "{batch_quote(mixed_wav_path)}" "{batch_quote(mic_wav_path)}"
if errorlevel 1 (
    echo.
    echo ERROR: vocal isolation failed. See the output above - common causes
    echo are the demucs package not being installed ^(re-run
    echo Install_PogEngine.bat^) or running out of GPU memory on a very long
    echo VOD ^(try setting the VOCAL_ISOLATION_SEGMENT_SECONDS environment
    echo variable - see pipeline_config.py^).
    if not "%RUN_ALL%"=="1" pause
    exit /b 1
)
del "{batch_quote(mixed_wav_path)}" >nul 2>nul
echo.
echo Done! Isolated voice saved as: {mic_wav_name}
echo You can now drag {mic_wav_name} onto 2_TranscribeAudio.bat
if not "%RUN_ALL%"=="1" pause
exit /b 0
'''

def make_extract_mic_bat(target_folder: Path, base_name: str, video_suffix: str, is_single_track: bool) -> str:
    """Dispatches to the multi-track (separate game/mic tracks) or
    single-track (Twitch-style merged track, needs vocal isolation) variant
    based on what count_audio_streams() found in organize_video()."""
    if is_single_track:
        return make_extract_mic_bat_singletrack(target_folder, base_name, video_suffix, ISOLATE_VOCALS_SCRIPT)
    return make_extract_mic_bat_multitrack(target_folder, base_name, video_suffix)

def make_transcribe_bat() -> str:
    return f'''@echo off
if "%~1"=="" (
    echo Drag your _mic.wav file onto this script to transcribe it.
    if not "%RUN_ALL%"=="1" pause
    exit /b 1
)

REM -----------------------------------------------------------------------
REM EDIT THESE THREE PATHS TO MATCH YOUR WHISPER.CPP INSTALL
set WHISPER_CLI={WHISPER_CLI}
set WHISPER_MODEL={WHISPER_MODEL}
set WHISPER_VAD={WHISPER_VAD}
REM -----------------------------------------------------------------------

set AUDIO=%~1
set OUTDIR=%~dp1
set AUDIO_NAME=%~n1

echo Transcribing %AUDIO_NAME% using whisper.cpp...
echo Model: %WHISPER_MODEL%
echo VAD model: %WHISPER_VAD%
echo Output folder: %OUTDIR%
echo.

REM No -ml/--max-len here on purpose: it forces every raw segment to that
REM many characters regardless of where you actually paused, which is what
REM produced choppy, mid-sentence-feeling subtitles (and whisper.cpp has a
REM known bug where -ml + -sow together can emit 0-duration "empty"
REM segments - see ggml-org/whisper.cpp#1967). Segmentation is left to VAD
REM (the --vad-* flags below), which breaks at real pauses in your speech
REM instead of an arbitrary character count. The natural-length segments
REM this produces then get split into short display captions (and
REM re-merged into fuller chunks for the LLM) by 3_FixSRT.bat and
REM 4_SplitSRT.bat - see split_sentence_into_natural_chunks() and
REM merge_entries_for_analysis() in this script.
REM
REM -mc -1 (unlimited context, whisper.cpp's own default) instead of a
REM small fixed number, so it keeps remembering earlier context across a
REM long stream instead of "forgetting" a few segments back.
"%WHISPER_CLI%" ^
  -m "%WHISPER_MODEL%" ^
  -f "%AUDIO%" ^
  -l en ^
  -osrt ^
  -of "%OUTDIR%%AUDIO_NAME%" ^
  -mc -1 ^
  --beam-size 5 ^
  --best-of 5 ^
  --entropy-thold 2.6 ^
  --logprob-thold -0.8 ^
  --no-speech-thold 0.7 ^
  --suppress-nst ^
  --vad ^
  -vm "%WHISPER_VAD%" ^
  --vad-threshold 0.42 ^
  --vad-min-silence-duration-ms 500 ^
  --vad-max-speech-duration-s 30 ^
  --vad-speech-pad-ms 200 ^
  -t 16

if errorlevel 1 (
    echo.
    echo ERROR: whisper.cpp transcription failed.
    echo Check that WHISPER_CLI and WHISPER_MODEL paths are correct inside this bat file.
    echo Open this file in Notepad and update the paths at the top.
    if not "%RUN_ALL%"=="1" pause
    exit /b 1
)

echo.
echo Done! SRT saved as: %OUTDIR%%AUDIO_NAME%.srt
echo Next: drag the .srt file onto 3_FixSRT.bat
if not "%RUN_ALL%"=="1" pause
exit /b 0
'''

def make_fix_srt_bat(script_path: Path) -> str:
    return f'''@echo off
if "%~1"=="" (
    echo Drag your .srt file onto this script.
    if not "%RUN_ALL%"=="1" pause
    exit /b 1
)

set SRT=%~1
set FOLDER=%~dp1

echo Fixing SRT timestamps and adjacent repeats: %~nx1
echo.
python "{batch_quote(script_path)}" --fix-srt "%SRT%" --no-pause

if errorlevel 1 (
    echo.
    echo ERROR: SRT fix failed.
    if not "%RUN_ALL%"=="1" pause
    exit /b 1
)

echo.
echo Done! Fixed SRT saved next to the original as *_fixed.srt.
echo Next: drag the *_fixed.srt file onto 4_SplitSRT.bat
if not "%RUN_ALL%"=="1" pause
exit /b 0
'''

def make_split_srt_bat(script_path: Path) -> str:
    return f'''@echo off
if "%~1"=="" (
    echo Drag your fixed .srt file onto this script.
    if not "%RUN_ALL%"=="1" pause
    exit /b 1
)

set SRT=%~1
set FOLDER=%~dp1

echo Splitting SRT into transcript_part files: %~nx1
echo.
python "{batch_quote(script_path)}" --split-srt "%SRT%" --no-pause

if errorlevel 1 (
    echo.
    echo ERROR: SRT split failed.
    if not "%RUN_ALL%"=="1" pause
    exit /b 1
)

echo.
echo Done! transcript_part files are ready in the folder.
echo Double-click 5_AnalyzeHighlights.bat when you are ready to find highlights
echo (or 6_RunAllSteps.bat to run steps 1 through 5 automatically).
if not "%RUN_ALL%"=="1" pause
exit /b 0
'''

def make_analyze_bat(target_folder: Path) -> str:
    """Step 5, the main entry point RunAll uses. Calls the merged analyzer
    with no --stage flag, so it walks discovery -> audioscan -> emotion ->
    verify -> judge -> export in order, skipping any stage whose checkpoint
    already exists. If it dies partway through, running this again (or
    RunAllSteps.bat) picks up exactly where it stopped."""
    return f'''@echo off
echo Running highlight analyzer on "{batch_quote(target_folder)}"...
echo.
python -u "{batch_quote(ANALYZE_HIGHLIGHTS)}" "{batch_quote(target_folder)}"
if errorlevel 1 (
    echo.
    echo ERROR: highlight analyzer failed.
    if not "%RUN_ALL%"=="1" pause
    exit /b 1
)
if not "%RUN_ALL%"=="1" pause
exit /b 0
'''

def make_debug_stage_bat(stage_key: str, step_label: str) -> str:
    """5a-5f: force-reruns exactly one internal stage, for debugging (e.g.
    after tweaking a prompt). NOT part of the main 1-6 sequence and not
    tracked by the RunAll GUI - double-click these directly when you want
    to. Forcing a stage clears every checkpoint after it, since they'd
    otherwise be stale leftovers from before the change."""
    return f'''@echo off
echo Force-running {step_label} on "%~dp0" (debug - clears later checkpoints)...
echo.
python -u "{batch_quote(ANALYZE_HIGHLIGHTS)}" "%~dp0" --stage {stage_key}
if errorlevel 1 (
    echo.
    echo ERROR: {step_label} failed.
    pause
    exit /b 1
)
pause
exit /b 0
'''

def make_run_all_bat(target_folder: Path, base_name: str, script_path: Path) -> str:
    return f'''@echo off
python "{batch_quote(script_path)}" --run-all-gui "{batch_quote(target_folder)}" --base-name "{batch_quote(base_name)}" --no-pause
if errorlevel 1 (
    echo.
    echo ERROR: Step 6 GUI failed to start.
    pause
    exit /b 1
)
exit /b 0
'''

@dataclass(frozen=True)
class RunAllStep:
    label: str
    bat_name: str
    input_kind: str | None = None
    pass_input: bool = False
    expected_kind: str | None = None

def build_run_all_steps(target_folder: Path, base_name: str) -> list[RunAllStep]:
    return [
        RunAllStep("1. Extract mic audio", "1_ExtractMicAudio.bat", expected_kind="mic_wav"),
        RunAllStep("2. Transcribe audio", "2_TranscribeAudio.bat", "mic_wav", True, "raw_srt"),
        RunAllStep("3. Fix SRT", "3_FixSRT.bat", "raw_srt", True, "fixed_srt"),
        RunAllStep("4. Split SRT", "4_SplitSRT.bat", "fixed_srt", True, "transcript_part"),
        # One external process from the GUI's view, but internally walks 6
        # checkpointed sub-stages (discovery -> ... -> export) with its own
        # resume logic - see analyze_highlights_emotion.py. "Done" here means
        # the final CSV exists; the script decides what still needs to run.
        RunAllStep("5. Analyze highlights", "5_AnalyzeHighlights.bat", "transcript_part", False, "highlights_csv"),
    ]

def newest_file(paths: Iterable[Path]) -> Path | None:
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return None
    return max(existing_paths, key=lambda path: path.stat().st_mtime)

def find_run_all_file(target_folder: Path, base_name: str, kind: str) -> Path | None:
    if kind == "mic_wav":
        exact_path = target_folder / f"{base_name}_mic.wav"
        return exact_path if exact_path.exists() else newest_file(target_folder.glob("*_mic.wav"))
    if kind == "raw_srt":
        exact_path = target_folder / f"{base_name}_mic.srt"
        candidates = [path for path in target_folder.glob("*.srt") if not path.stem.casefold().endswith("_fixed")]
        return exact_path if exact_path.exists() else newest_file(candidates)
    if kind == "fixed_srt":
        exact_path = target_folder / f"{base_name}_mic_fixed.srt"
        return exact_path if exact_path.exists() else newest_file(target_folder.glob("*_fixed.srt"))
    if kind == "transcript_part":
        exact_path = target_folder / "transcript_part1.txt"
        return exact_path if exact_path.exists() else newest_file(target_folder.glob("transcript_part*.txt"))
    if kind == "highlights_csv":
        return newest_file(target_folder.glob("top*_highlights.csv"))
    raise ValueError(f"Unknown run-all file kind: {kind}")

def describe_run_all_file(target_folder: Path, base_name: str, kind: str) -> str:
    if kind == "mic_wav":
        return f"{target_folder / f'{base_name}_mic.wav'} or newest *_mic.wav"
    if kind == "raw_srt":
        return f"{target_folder / f'{base_name}_mic.srt'} or newest non-fixed *.srt"
    if kind == "fixed_srt":
        return f"{target_folder / f'{base_name}_mic_fixed.srt'} or newest *_fixed.srt"
    if kind == "transcript_part":
        return f"{target_folder / 'transcript_part1.txt'} or newest transcript_part*.txt"
    if kind == "highlights_csv":
        return f"{target_folder} / top*_highlights.csv"
    raise ValueError(f"Unknown run-all file kind: {kind}")

def gallery_image_paths(gallery_dir: Path = GALLERY_DIR) -> list[Path]:
    if not gallery_dir.exists():
        return []

    image_paths = [
        path
        for path in gallery_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in GALLERY_IMAGE_EXTENSIONS
    ]
    return sorted(image_paths, key=lambda path: path.stat().st_mtime, reverse=True)

def make_step6_log_path(target_folder: Path) -> Path:
    return target_folder / f"step6_run_{datetime.now():%Y%m%d_%H%M%S}.log"

def run_all_gui(target_folder: Path, base_name: str) -> int:
    target_folder = target_folder.resolve()
    steps = build_run_all_steps(target_folder, base_name)

    import tkinter as tk
    from tkinter import ttk

    try:
        from PIL import Image, ImageTk
    except ImportError:
        Image = None
        ImageTk = None

    events: queue.Queue[tuple[str, object]] = queue.Queue()
    run_log_path = make_step6_log_path(target_folder)
    run_log = run_log_path.open("w", encoding="utf-8", buffering=1)
    run_log_closed = {"value": False}
    run_log.write("Step 6 run log\n")
    run_log.write(f"Started: {datetime.now().isoformat(timespec='seconds')}\n")
    run_log.write(f"Folder: {target_folder}\n")
    run_log.write(f"Base name: {base_name}\n\n")

    # Stop-button state: current step's subprocess (so Stop can kill it),
    # whether a stop was requested (so a killed step reports "Stopped" not
    # "Failed"), and the last fully-completed step for the status bar.
    current_process: dict[str, subprocess.Popen | None] = {"popen": None}
    stop_requested = {"value": False}

    def write_run_log(text: str) -> None:
        if run_log_closed["value"]:
            return
        run_log.write(text)
        run_log.flush()

    def close_run_log() -> None:
        if run_log_closed["value"]:
            return
        run_log.write(f"\nClosed: {datetime.now().isoformat(timespec='seconds')}\n")
        run_log.close()
        run_log_closed["value"] = True

    root = tk.Tk()
    root.title("Step 6 - Run All VOD Steps - Emotion Edition")
    root.geometry("980x640")
    root.configure(bg="#121212")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background="#121212", foreground="#f2f2f2", fieldbackground="#1e1e1e")
    style.configure("TFrame", background="#121212")
    style.configure("TLabelframe", background="#121212", foreground="#f2f2f2")
    style.configure("TLabelframe.Label", background="#121212", foreground="#f2f2f2")
    style.configure("TLabel", background="#121212", foreground="#f2f2f2")
    style.configure("TButton", background="#242424", foreground="#f2f2f2", bordercolor="#3a3a3a")
    style.map("TButton", background=[("active", "#333333")])
    style.configure(
        "Orange.Horizontal.TProgressbar",
        troughcolor="#2a2a2a",
        background="#ff8c00",
        lightcolor="#ff8c00",
        darkcolor="#ff8c00",
        bordercolor="#2a2a2a",
    )
    style.configure("Stop.TButton", background="#5a1f1f", foreground="#f2f2f2", bordercolor="#7a2a2a")
    style.map("Stop.TButton", background=[("active", "#7a2a2a"), ("disabled", "#3a2626")])

    status_var = tk.StringVar(value=f"Ready to run. Log: {run_log_path}")
    progress_var = tk.DoubleVar(value=0)
    exit_code = {"value": 0}

    main_frame = ttk.Frame(root, padding=12)
    main_frame.pack(fill="both", expand=True)
    main_frame.columnconfigure(0, weight=3)
    main_frame.columnconfigure(1, weight=2)
    main_frame.rowconfigure(2, weight=1)

    ttk.Label(main_frame, text="Step 6: Run everything + emotion analysis", font=("Segoe UI", 16, "bold")).grid(
        row=0, column=0, sticky="w"
    )

    status_row = ttk.Frame(main_frame)
    status_row.grid(row=1, column=0, sticky="ew", pady=(4, 8))
    status_row.columnconfigure(0, weight=1)
    ttk.Label(status_row, textvariable=status_var).grid(row=0, column=0, sticky="w")
    stop_button = ttk.Button(status_row, text="Stop", style="Stop.TButton")
    stop_button.grid(row=0, column=1, sticky="e", padx=(10, 0))

    step_frame = ttk.LabelFrame(main_frame, text="Progress", padding=10)
    step_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 10))
    step_status_vars: list[tk.StringVar] = []
    for row, step in enumerate(steps):
        ttk.Label(step_frame, text=step.label).grid(row=row, column=0, sticky="w", pady=3)
        state_var = tk.StringVar(value="Waiting")
        step_status_vars.append(state_var)
        ttk.Label(step_frame, textvariable=state_var, width=18).grid(row=row, column=1, sticky="e", pady=3)

    progress = ttk.Progressbar(main_frame, variable=progress_var, maximum=len(steps), style="Orange.Horizontal.TProgressbar")
    progress.grid(row=3, column=0, sticky="ew", padx=(0, 10), pady=(10, 0))

    log_box = tk.Text(main_frame, height=12, wrap="word", state="disabled", bg="#0f0f0f", fg="#f2f2f2", insertbackground="#f2f2f2")
    log_box.grid(row=4, column=0, sticky="nsew", padx=(0, 10), pady=(10, 0))

    gallery_frame = ttk.LabelFrame(main_frame, text="Best Of Gallery", padding=4)
    gallery_frame.grid(row=0, column=1, rowspan=5, sticky="nsew")
    gallery_frame.rowconfigure(0, weight=1)
    gallery_frame.columnconfigure(0, weight=1)

    image_label = ttk.Label(gallery_frame, text="Loading gallery images...", anchor="center")
    image_label.grid(row=0, column=0, columnspan=2, sticky="nsew")

    gallery_paths = gallery_image_paths()
    gallery_index = {"value": 0}
    gallery_photo = {"value": None}
    current_gallery_path = {"value": None}

    def append_log(text: str) -> None:
        log_box.configure(state="normal")
        log_box.insert("end", text)
        log_box.see("end")
        log_box.configure(state="disabled")
        write_run_log(text)

    def render_gallery_image(image_path: Path) -> None:
        try:
            height = max(gallery_frame.winfo_height() - 54, image_label.winfo_height(), 360)
            if Image is not None and ImageTk is not None:
                with Image.open(image_path) as source_image:
                    aspect_ratio = source_image.width / max(source_image.height, 1)
                    width = max(round(height * aspect_ratio), 1)
                    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                    image = source_image.resize((width, height), resample)
                gallery_photo["value"] = ImageTk.PhotoImage(image)
            else:
                gallery_photo["value"] = tk.PhotoImage(file=str(image_path))
            image_label.configure(image=gallery_photo["value"], text="")
        except Exception as exc:
            image_label.configure(text=f"Could not load gallery image:\n{exc}", image="")
            gallery_photo["value"] = None

    def show_gallery_image(offset: int = 0) -> None:
        if not gallery_paths:
            image_label.configure(text=f"No images found in:\n{GALLERY_DIR}", image="")
            return

        gallery_index["value"] = (gallery_index["value"] + offset) % len(gallery_paths)

        current_gallery_path["value"] = gallery_paths[gallery_index["value"]]
        render_gallery_image(current_gallery_path["value"])

    ttk.Button(gallery_frame, text="Previous", command=lambda: show_gallery_image(-1)).grid(
        row=1, column=0, sticky="ew", pady=(8, 0), padx=(0, 4)
    )
    ttk.Button(gallery_frame, text="Next", command=lambda: show_gallery_image(1)).grid(
        row=1, column=1, sticky="ew", pady=(8, 0), padx=(4, 0)
    )

    def rerender_gallery_image(_event: object | None = None) -> None:
        if current_gallery_path["value"] is not None:
            render_gallery_image(current_gallery_path["value"])

    image_label.bind("<Configure>", rerender_gallery_image)

    def rotate_gallery() -> None:
        show_gallery_image(1)
        root.after(8000, rotate_gallery)

    def run_step_process(index: int, step: RunAllStep) -> bool:
        step_path = target_folder / step.bat_name
        if not step_path.exists():
            events.put(("failed", f"Missing step script: {step_path}"))
            return False

        if step.expected_kind is not None:
            existing_output = find_run_all_file(target_folder, base_name, step.expected_kind)
            if existing_output is not None:
                events.put(("status", (index, "Skipped")))
                events.put(("log", f"\n--- {step.label} ---\n"))
                events.put(("log", f"Skipping because output already exists: {existing_output}\n"))
                events.put(("progress", index + 1))
                return True

        events.put(("status", (index, "Running")))
        events.put(("log", f"\n--- {step.label} ---\n"))

        args: list[str] = []
        if step.input_kind is not None:
            input_path = find_run_all_file(target_folder, base_name, step.input_kind)
            if input_path is None:
                events.put(("status", (index, "Failed")))
                events.put(("failed", f"{step.label} could not find input file:\n{describe_run_all_file(target_folder, base_name, step.input_kind)}"))
                return False
            events.put(("log", f"Using input: {input_path}\n"))
            if step.pass_input:
                args.append(str(input_path))

        command = ["cmd.exe", "/c", "call", str(step_path), *args]
        env = os.environ.copy()
        env["RUN_ALL"] = "1"
        events.put(("log", f"Command: {' '.join(command)}\n"))

        process = subprocess.Popen(
            command,
            cwd=target_folder,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        current_process["popen"] = process

        assert process.stdout is not None
        for line in process.stdout:
            events.put(("log", line))

        return_code = process.wait()
        current_process["popen"] = None

        if stop_requested["value"]:
            events.put(("status", (index, "Stopped")))
            return False

        if return_code:
            events.put(("status", (index, "Failed")))
            events.put(("failed", f"{step.label} failed with exit code {return_code}."))
            return False

        if step.expected_kind is not None:
            expected_output = find_run_all_file(target_folder, base_name, step.expected_kind)
            if expected_output is None:
                events.put(("status", (index, "Failed")))
                events.put(("failed", f"{step.label} did not create expected file:\n{describe_run_all_file(target_folder, base_name, step.expected_kind)}"))
                return False
            events.put(("log", f"Found output: {expected_output}\n"))

        events.put(("status", (index, "Done")))
        events.put(("progress", index + 1))
        return True

    def kill_process_tree(process: subprocess.Popen | None) -> None:
        if process is None or process.poll() is not None:
            return  # nothing running, or it already exited on its own
        try:
            # taskkill /T kills the whole tree, not just cmd.exe - Popen.
            # terminate() alone only signals cmd.exe, which won't reliably
            # take python.exe (and whatever it's waiting on) down with it.
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True, timeout=15,
            )
            events.put(("log", f"[stop] Killed process tree (PID {process.pid}).\n"))
        except Exception as exc:
            events.put(("log", f"[stop] taskkill failed: {exc}\n"))

    def unload_ollama_models_now() -> None:
        for model_name in (MODEL, JUDGE_MODEL):
            try:
                requests.post(
                    OLLAMA_URL,
                    json={"model": model_name, "prompt": "", "keep_alive": 0},
                    timeout=10,
                )
                events.put(("log", f"[stop] Requested Ollama unload for {model_name}.\n"))
            except Exception as exc:
                events.put(("log", f"[stop] Could not unload {model_name}: {exc}\n"))

    def clear_in_progress_temp_files() -> None:
        """Atomic checkpoint writes (write-to-.tmp, then rename) mean a
        killed process can never leave a half-written checkpoint - the
        finished file either exists or it doesn't. The only possible
        leftover is a .tmp that was written but not yet renamed; clean
        those up so a stray file doesn't sit in the folder."""
        removed = []
        for pattern in ("checkpoint_*.json.tmp", "pipeline_stats.json.tmp"):
            for path in target_folder.glob(pattern):
                try:
                    path.unlink()
                    removed.append(path.name)
                except OSError:
                    pass
        if removed:
            events.put(("log", f"[stop] Removed in-progress temp file(s): {', '.join(removed)}\n"))

    def determine_last_completed_step() -> str:
        last_done_label = "none yet"
        for step in steps:
            if step.expected_kind is None:
                continue
            if find_run_all_file(target_folder, base_name, step.expected_kind) is not None:
                last_done_label = step.label
        return last_done_label

    def on_stop_clicked() -> None:
        if stop_requested["value"]:
            return  # already stopping, ignore extra clicks
        stop_requested["value"] = True
        stop_button.configure(state="disabled", text="Stopping...")
        events.put(("log", "\n--- STOP requested ---\n"))

        def stop_worker() -> None:
            # Background thread so a slow taskkill or unresponsive Ollama
            # can't freeze the GUI - cross-thread comms go through the
            # events queue, same as the main worker() thread.
            kill_process_tree(current_process["popen"])
            unload_ollama_models_now()
            clear_in_progress_temp_files()

            last_done = determine_last_completed_step()
            stop_info = {
                "stopped_at": datetime.now().isoformat(timespec="seconds"),
                "last_completed_step": last_done,
            }
            try:
                with open(target_folder / "last_stop_state.json", "w", encoding="utf-8") as f:
                    json.dump(stop_info, f, indent=2)
                events.put(("log", f"[stop] Logged last_stop_state.json (last completed: {last_done}).\n"))
            except Exception as exc:
                events.put(("log", f"[stop] Could not write last_stop_state.json: {exc}\n"))

            events.put(("stopped", last_done))

        threading.Thread(target=stop_worker, daemon=True).start()

    stop_button.configure(command=on_stop_clicked)

    def worker() -> None:
        for index, step in enumerate(steps):
            if not run_step_process(index, step):
                return
        events.put(("complete", "All steps completed successfully."))

    def drain_events() -> None:
        while True:
            try:
                kind, payload = events.get_nowait()
            except queue.Empty:
                break

            if kind == "status":
                index, value = payload
                step_status_vars[index].set(value)
                status_var.set(f"{steps[index].label}: {value}")
                write_run_log(f"[STATUS] {steps[index].label}: {value}\n")
            elif kind == "progress":
                progress_var.set(payload)
            elif kind == "log":
                append_log(payload)
            elif kind == "failed":
                exit_code["value"] = 1
                status_var.set(str(payload))
                append_log(f"\nERROR: {payload}\n")
                stop_button.configure(state="disabled")
            elif kind == "stopped":
                last_done = payload
                status_var.set(
                    f"Stopped by user. Last fully completed step: {last_done}. "
                    f"Run 6_RunAllSteps.bat again to continue from there."
                )
                append_log(
                    f"\nStopped by user. Last fully completed step: {last_done}.\n"
                    "Ollama models unloaded. Run 6_RunAllSteps.bat again to continue.\n"
                )
                stop_button.configure(state="disabled", text="Stopped")
            elif kind == "complete":
                status_var.set(str(payload))
                append_log(f"\n{payload}\n")
                progress_var.set(len(steps))
                write_run_log(f"[PROGRESS] {len(steps)}/{len(steps)}\n")
                stop_button.configure(state="disabled")

        root.after(100, drain_events)

    show_gallery_image()
    if len(gallery_paths) > 1:
        root.after(8000, rotate_gallery)

    def on_window_close() -> None:
        close_run_log()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_window_close)

    threading.Thread(target=worker, daemon=True).start()
    root.after(100, drain_events)
    root.mainloop()
    close_run_log()
    return exit_code["value"]

def move_related_files(video_file: Path, target_folder: Path) -> int:
    base_name = video_file.stem.lower()
    moved_count = 0

    for item in video_file.parent.iterdir():
        if item.resolve() == target_folder.resolve():
            continue
        if item.stem.lower().startswith(base_name):
            destination = target_folder / item.name
            if item.resolve() == destination.resolve():
                continue
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(item), str(destination))
            moved_count += 1

    return moved_count

def organize_video(video_file: Path) -> Path:
    video_file = video_file.resolve()
    if not video_file.exists():
        raise FileNotFoundError(f"Input file not found: {video_file}")

    base_name = video_file.stem
    target_folder = video_file.parent / base_name
    target_folder.mkdir(exist_ok=True)

    moved_count = move_related_files(video_file, target_folder)
    print(f"Moved {moved_count} file(s) to '{target_folder}'")

    video_suffix = video_file.suffix or ".mp4"
    moved_video_path = target_folder / video_file.name
    stream_count = count_audio_streams(moved_video_path)

    if stream_count is None:
        is_single_track = False
        detection_note = "ffprobe unavailable or probe failed - assumed multi-track"
        print("[!] Could not detect audio track count (ffprobe missing, or the probe failed).")
        print("    Assuming a locally recorded VOD with separate game/mic tracks (2+).")
        print("    If this is actually a single-track Twitch VOD, install ffprobe (it ships")
        print("    with ffmpeg) and re-run, or edit 1_ExtractMicAudio.bat by hand.")
    elif stream_count <= 1:
        is_single_track = True
        detection_note = f"{stream_count} audio stream(s) detected"
        print(f"Detected {stream_count} audio track in {moved_video_path.name} - this looks like a")
        print("Twitch-style VOD with everything mixed into one track. 1_ExtractMicAudio.bat will")
        print("isolate the streamer's voice from game audio/music/alerts automatically.")
    else:
        is_single_track = False
        detection_note = f"{stream_count} audio stream(s) detected"
        print(f"Detected {stream_count} audio tracks in {moved_video_path.name} - treating this as a")
        print("locally recorded VOD with a separate mic track (track index 1).")

    try:
        with open(target_folder / "vod_audio_info.json", "w", encoding="utf-8") as f:
            json.dump({
                "source_video": moved_video_path.name,
                "audio_stream_count": stream_count,
                "single_track_mode": is_single_track,
                "detection_note": detection_note,
                "detected_at": datetime.now().isoformat(timespec="seconds"),
            }, f, indent=2)
    except OSError as exc:
        print(f"[!] Could not write vod_audio_info.json: {exc}")

    script_path = Path(__file__).resolve()
    bat_files = {
        "1_ExtractMicAudio.bat": make_extract_mic_bat(target_folder, base_name, video_suffix, is_single_track),
        "2_TranscribeAudio.bat": make_transcribe_bat(),
        "3_FixSRT.bat": make_fix_srt_bat(script_path),
        "4_SplitSRT.bat": make_split_srt_bat(script_path),
        "5_AnalyzeHighlights.bat": make_analyze_bat(target_folder),
        "6_RunAllSteps.bat": make_run_all_bat(target_folder, base_name, script_path),
        # Debug-only sub-steps: not in the main numbered sequence or tracked
        # by the RunAll GUI. Force-rerun one internal stage in isolation
        # (e.g. after tweaking a prompt) without redoing everything before it.
        "5a_Discovery.bat": make_debug_stage_bat("discovery", "Discovery"),
        "5b_AudioScan.bat": make_debug_stage_bat("audioscan", "Audio Scan"),
        "5c_EmotionScoring.bat": make_debug_stage_bat("emotion", "Emotion Scoring"),
        "5d_Verify.bat": make_debug_stage_bat("verify", "Verification"),
        "5e_Judge.bat": make_debug_stage_bat("judge", "Judging"),
        "5f_Export.bat": make_debug_stage_bat("export", "Export"),
    }

    for name, content in bat_files.items():
        write_text_crlf(target_folder / name, content, encoding="ascii")

    print(f"Created helper scripts in '{target_folder}':")
    if is_single_track:
        print("   1_ExtractMicAudio.bat   <- double-click to isolate the streamer's voice")
        print("                              (single-track Twitch-style VOD)")
    else:
        print("   1_ExtractMicAudio.bat   <- double-click to extract the mic track")
        print("                              (separate-track local recording)")
    print("   2_TranscribeAudio.bat   <- drag _mic.wav onto this to transcribe")
    print("   3_FixSRT.bat            <- drag whisper .srt onto this to fix repeats/timestamps")
    print("   4_SplitSRT.bat          <- drag *_fixed.srt onto this to create transcript_part files")
    print("   5_AnalyzeHighlights.bat <- double-click for emotion-enhanced highlights")
    print("   6_RunAllSteps.bat       <- double-click to run steps 1 through 5 in order")
    print("\n   Debug only (not part of the main sequence, not tracked by RunAll):")
    print("   5a_Discovery.bat        <- force-rerun just the LLM discovery passes")
    print("   5b_AudioScan.bat        <- force-rerun just the full-file audio scan")
    print("   5c_EmotionScoring.bat   <- force-rerun just the speech-emotion model")
    print("   5d_Verify.bat           <- force-rerun just the content verification pass")
    print("   5e_Judge.bat            <- force-rerun just the final ranking")
    print("   5f_Export.bat           <- rewrite the CSV + Resolve EDL from the last judged result")
    print("\nStep 5 checkpoints internally after each of its sub-stages, so if it dies or")
    print("you hit Stop in the RunAll GUI partway through, running it again (or")
    print("6_RunAllSteps.bat) picks up exactly where it stopped instead of starting over.")
    print("Forcing a debug sub-step (5a-5f) clears any later checkpoints, since they'd")
    print("otherwise be stale leftovers from before whatever you changed.")
    print("\nDone.")
    return target_folder

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize an OBS VOD folder, create helper batch files, fix/split Whisper SRT files, and run emotion-enhanced highlight analysis."
    )
    parser.add_argument("video_file", nargs="?", help="Video file to organize. Drag an OBS .mp4 onto this script or launcher.")
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument("--fix-srt", metavar="SRT", help="Fix a Whisper .srt file without splitting it.")
    action_group.add_argument("--split-srt", metavar="SRT", help="Split a fixed .srt file into transcript_part files.")
    action_group.add_argument("--run-all-gui", metavar="FOLDER", help="Open the step 6 GUI runner for an organized VOD folder.")
    parser.add_argument("--base-name", help="Video base name for --run-all-gui. Defaults to the folder name.")
    parser.add_argument("--chunk-minutes", type=int, default=DEFAULT_CHUNK_MINUTES, help="Transcript chunk length for --split-srt.")
    parser.add_argument("--no-pause", action="store_true", help="Do not wait for Enter before exiting.")
    return parser.parse_args(argv)

def pause_if_needed(enabled: bool) -> None:
    if enabled:
        try:
            input("Press Enter to continue...")
        except EOFError:
            pass

def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    should_pause = not args.no_pause

    try:
        if args.fix_srt:
            fix_srt(Path(args.fix_srt))
            return 0

        if args.split_srt:
            split_srt_into_chunks(Path(args.split_srt), args.chunk_minutes)
            return 0

        if args.run_all_gui:
            target_folder = Path(args.run_all_gui)
            base_name = args.base_name or target_folder.name
            return run_all_gui(target_folder, base_name)

        if not args.video_file:
            print("No input file provided. Drag a video file onto this script or pass the path as an argument.")
            return 1

        organize_video(Path(args.video_file))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        pause_if_needed(should_pause)

if __name__ == "__main__":
    raise SystemExit(main())