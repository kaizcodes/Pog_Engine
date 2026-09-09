from __future__ import annotations

import argparse
import csv
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
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from pipeline_config import (
    AUTO_SLEEP_AFTER_PIPELINE,
    STEP_FOLDER_NAMES,
    step_subdir,
    AUTO_SLEEP_DELAY_SECONDS,
    BIG_STEP_LABELS,
    EMOTION_ENABLED,
    EMOTION_MODEL_ID,
    JUDGE_MODEL,
    MODEL,
    LLM_BACKEND,
    LLAMA_SERVER_URL, LLAMA_MODEL_PATH, LLAMA_DISCOVERY_MODEL_PATH, LLAMA_JUDGE_MODEL_PATH, LLAMA_CONTEXT_SIZE,
    NOISE_GATE_ATTACK_MS,
    NOISE_GATE_RATIO,
    NOISE_GATE_RELEASE_MS,
    NOISE_GATE_THRESHOLD_DB,
    OLLAMA_URL,
    STEP_HISTORY_FILENAME,
    TRANSCRIPTION_CHUNK_MINUTES,
    TRANSCRIPTION_CHUNK_OVERLAP_SECONDS,
    VOCAL_ISOLATION_MODEL,
    ollama_base_url,
    ollama_is_reachable,
    ollama_not_ready_message,
)

DEFAULT_CHUNK_MINUTES = 25
MIN_REPEAT_SENTENCE_WORDS = 3
MIN_SUBTITLE_WORDS = 3
MAX_SUBTITLE_LINE_CHARS = 42
# Maximum characters per displayed SRT line; Whisper controls caption boundaries.
# Words per "thought" for LLM discovery to read (see merge_entries_for_analysis())
# - independent of the on-screen caption line width above.
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
def probe_audio_duration_seconds(audio_path: Path) -> float:
    """Return the input duration using ffprobe, or raise a useful error."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is not on PATH; install ffmpeg and retry.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffprobe timed out while reading {audio_path.name}.") from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown ffprobe error"
        raise RuntimeError(f"Could not read audio duration for {audio_path.name}: {detail}")

    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"ffprobe returned an invalid duration for {audio_path.name}: {result.stdout.strip()!r}"
        ) from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"Audio duration is invalid for {audio_path.name}: {duration!r}")
    return duration


def _transcribe_audio_chunk(
    chunk_audio_path: Path,
    chunk_srt_path: Path,
) -> None:
    """Run one fresh whisper.cpp process and require its SRT output."""
    chunk_srt_path.unlink(missing_ok=True)
    command = [
        str(WHISPER_CLI),
        "-m", str(WHISPER_MODEL),
        "-f", str(chunk_audio_path),
        "-l", "en",
        "-osrt",
        "-of", str(chunk_srt_path.with_suffix("")),
        "-mc", "-1",
        "--beam-size", "5",
        "--best-of", "5",
        "--entropy-thold", "2.6",
        "--logprob-thold", "-0.8",
        "--no-speech-thold", "0.7",
        "--suppress-nst",
        "--vad",
        "-vm", str(WHISPER_VAD),
        "--vad-threshold", "0.42",
        "--vad-min-silence-duration-ms", "500",
        "--vad-max-speech-duration-s", "30",
        "--vad-speech-pad-ms", "200",
        "-t", "16",
    ]
    result = subprocess.run(command)
    if result.returncode != 0:
        raise RuntimeError(
            f"whisper.cpp failed for {chunk_audio_path.name} with exit code {result.returncode}."
        )
    if not chunk_srt_path.is_file():
        raise RuntimeError(
            f"whisper.cpp completed but did not create {chunk_srt_path.name}."
        )


def _read_transcription_chunk(
    chunk_srt_path: Path,
    offset_ms: int,
    chunk_index: int,
) -> list[tuple[int, int, str, int]]:
    """Read one chunk SRT and shift its timestamps into the full-audio clock."""
    entries: list[tuple[int, int, str, int]] = []
    for block in split_srt_blocks(read_text(chunk_srt_path)):
        lines = re.split(r"\r?\n", block)
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        start_text, end_text = [part.strip() for part in lines[1].split("-->", 1)]
        start_ms = parse_srt_time(start_text) + offset_ms
        end_ms = parse_srt_time(end_text) + offset_ms
        if end_ms <= start_ms:
            continue
        text = normalize_caption_text(lines[2:])
        if text:
            entries.append((start_ms, end_ms, text, chunk_index))
    return entries


def stitch_transcription_chunks(
    chunk_srt_paths: list[Path],
    chunk_offsets_ms: list[int],
    output_path: Path,
) -> Path:
    """Shift each chunk into the full-audio clock before writing one SRT."""
    if len(chunk_srt_paths) != len(chunk_offsets_ms):
        raise ValueError("Chunk SRT paths and timestamp offsets must have equal lengths.")
    if any(offset_ms < 0 for offset_ms in chunk_offsets_ms):
        raise ValueError("Chunk timestamp offsets cannot be negative.")

    entries: list[tuple[int, int, str, int]] = []
    for chunk_index, (chunk_path, offset_ms) in enumerate(
        zip(chunk_srt_paths, chunk_offsets_ms)
    ):
        entries.extend(_read_transcription_chunk(chunk_path, offset_ms, chunk_index))
    entries.sort(key=lambda item: (item[0], item[1], item[3]))

    deduped: list[list[int | str]] = []
    removed_overlap_count = 0
    for start_ms, end_ms, text, chunk_index in entries:
        if deduped:
            previous_start, previous_end, previous_text, previous_chunk = deduped[-1]
            same_overlap_caption = (
                chunk_index != previous_chunk
                and normalize_repeated_sentence_key(text)
                == normalize_repeated_sentence_key(str(previous_text))
                and start_ms <= int(previous_end) + 1_000
            )
            if same_overlap_caption:
                deduped[-1][0] = min(int(previous_start), start_ms)
                deduped[-1][1] = max(int(previous_end), end_ms)
                removed_overlap_count += 1
                continue
        deduped.append([start_ms, end_ms, text, chunk_index])

    if not deduped:
        raise RuntimeError("No subtitle entries were produced by the audio chunks.")
    for start_ms, end_ms, _text, _chunk_index in deduped:
        if int(start_ms) < 0 or int(end_ms) <= int(start_ms):
            raise RuntimeError("Stitching produced an invalid connected subtitle timestamp.")

    blocks = [
        make_srt_block(index, int(start_ms), int(end_ms), str(text))
        for index, (start_ms, end_ms, text, _chunk_index) in enumerate(deduped, start=1)
    ]
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    write_text_crlf(temporary_path, "\r\n\r\n".join(blocks), encoding="utf-8")
    temporary_path.replace(output_path)
    print(
        f"Stitched {len(blocks)} subtitle block(s) into {output_path.name}; "
        f"removed {removed_overlap_count} overlap duplicate(s)."
    )
    return output_path


def _chunk_manifest_path(chunk_dir: Path) -> Path:
    return chunk_dir / "chunk_manifest.json"


def prepare_audio_chunks(
    audio_path: Path,
    chunk_minutes: int = TRANSCRIPTION_CHUNK_MINUTES,
    overlap_seconds: int = TRANSCRIPTION_CHUNK_OVERLAP_SECONDS,
) -> Path:
    """Extract every overlapping audio chunk and persist its timing manifest."""
    audio_path = audio_path.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if chunk_minutes <= 0:
        raise ValueError("Transcription chunk length must be greater than zero minutes.")
    if overlap_seconds < 0:
        raise ValueError("Transcription chunk overlap cannot be negative.")

    chunk_seconds = chunk_minutes * 60
    if overlap_seconds >= chunk_seconds:
        raise ValueError("Transcription chunk overlap must be shorter than the chunk length.")

    duration_seconds = probe_audio_duration_seconds(audio_path)
    chunk_count = math.ceil(duration_seconds / chunk_seconds)
    chunk_dir = audio_path.parent / f"{audio_path.stem}_transcription_chunks"
    chunk_dir.mkdir(exist_ok=True)
    existing_manifest: dict[str, object] | None = None
    try:
        candidate = json.loads(_chunk_manifest_path(chunk_dir).read_text(encoding="utf-8"))
        if isinstance(candidate, dict):
            existing_manifest = candidate
    except (OSError, ValueError):
        pass
    reusable_manifest = (
        existing_manifest is not None
        and existing_manifest.get("source_audio") == audio_path.name
        and existing_manifest.get("duration_ms") == round(duration_seconds * 1_000)
        and existing_manifest.get("chunk_minutes") == chunk_minutes
        and existing_manifest.get("overlap_seconds") == overlap_seconds
        and isinstance(existing_manifest.get("chunks"), list)
        and len(existing_manifest["chunks"]) == chunk_count
    )
    manifest_entries: list[dict[str, object]] = []
    print(
        f"Preparing {audio_path.name} in {chunk_count} chunk(s) "
        f"({chunk_minutes} minutes each, {overlap_seconds}s overlap).",
        flush=True,
    )
    for chunk_number in range(chunk_count):
        nominal_start = chunk_number * chunk_seconds
        nominal_end = min(duration_seconds, nominal_start + chunk_seconds)
        extract_start = max(0.0, nominal_start - (overlap_seconds if chunk_number else 0))
        extract_end = min(
            duration_seconds,
            nominal_end + (overlap_seconds if nominal_end < duration_seconds else 0),
        )
        extract_duration = extract_end - extract_start
        chunk_audio_path = chunk_dir / f"{audio_path.stem}_chunk_{chunk_number + 1:04d}.wav"
        if reusable_manifest and chunk_audio_path.is_file() and chunk_audio_path.stat().st_size > 0:
            print(
                f"Using preserved audio chunk "
                f"{chunk_number + 1:02d}/{chunk_count:02d}: {chunk_audio_path.name}",
                flush=True,
            )
        else:
            print(
                f"\nExtracting chunk {chunk_number + 1:02d}/{chunk_count:02d}: "
                f"{format_plain_time(round(nominal_start * 1000))} - "
                f"{format_plain_time(round(nominal_end * 1000))}",
                flush=True,
            )
            ffmpeg_result = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-y",
                    "-ss", f"{extract_start:.3f}",
                    "-i", str(audio_path),
                    "-t", f"{extract_duration:.3f}",
                    "-map", "0:a:0",
                    "-c:a", "pcm_s16le",
                    str(chunk_audio_path),
                ]
            )
            if (
                ffmpeg_result.returncode != 0
                or not chunk_audio_path.is_file()
                or chunk_audio_path.stat().st_size <= 0
            ):
                raise RuntimeError(f"ffmpeg failed to create {chunk_audio_path.name}.")

        manifest_entries.append({
            "filename": chunk_audio_path.name,
            "offset_ms": round(extract_start * 1_000),
            "nominal_start_ms": round(nominal_start * 1_000),
            "nominal_end_ms": round(nominal_end * 1_000),
        })

    manifest = {
        "source_audio": audio_path.name,
        "duration_ms": round(duration_seconds * 1_000),
        "chunk_minutes": chunk_minutes,
        "overlap_seconds": overlap_seconds,
        "chunks": manifest_entries,
    }
    temporary_path = _chunk_manifest_path(chunk_dir).with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_path.replace(_chunk_manifest_path(chunk_dir))
    print(
        f"Finished preparing {len(manifest_entries)} audio chunk(s) in "
        f"{chunk_dir.name}/. Whisper can now run on each saved chunk.",
        flush=True,
    )
    return chunk_dir


def _load_prepared_audio_chunks(audio_path: Path) -> tuple[list[Path], list[int]]:
    chunk_dir = audio_path.parent / f"{audio_path.stem}_transcription_chunks"
    manifest_path = _chunk_manifest_path(chunk_dir)
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Prepared chunk manifest is missing: {manifest_path}. "
            "Run Step 1 completely before starting Step 2."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest["chunks"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Could not read prepared chunk manifest: {manifest_path}") from exc
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"Prepared chunk manifest contains no chunks: {manifest_path}")

    chunk_paths: list[Path] = []
    offsets_ms: list[int] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError(f"Invalid chunk entry in manifest: {manifest_path}")
        filename = entry.get("filename")
        offset_ms = entry.get("offset_ms")
        if not isinstance(filename, str) or not isinstance(offset_ms, int):
            raise RuntimeError(f"Invalid chunk timing entry in manifest: {manifest_path}")
        chunk_path = chunk_dir / filename
        if not chunk_path.is_file() or chunk_path.stat().st_size <= 0:
            raise RuntimeError(
                f"Prepared audio chunk is missing or empty: {chunk_path}. "
                "Run Step 1 again to rebuild the chunk set."
            )
        chunk_paths.append(chunk_path)
        offsets_ms.append(offset_ms)
    return chunk_paths, offsets_ms


def transcribe_audio_in_chunks(
    audio_path: Path,
) -> Path:
    """Run Whisper once per already-prepared chunk, then stitch all SRTs."""
    audio_path = audio_path.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    chunk_audio_paths, chunk_offsets_ms = _load_prepared_audio_chunks(audio_path)
    chunk_srt_paths: list[Path] = []
    print(
        f"Transcribing {audio_path.name} in {len(chunk_audio_paths)} saved chunk(s); "
        "all chunk audio is already prepared.",
        flush=True,
    )
    # The stitched SRT belongs to step 2's folder. When the mic wav lives in
    # step 1's folder (the RunAll layout), resolve the VOD root from it;
    # standalone/drag-drop runs keep the SRT next to the audio.
    if audio_path.parent.name == STEP_FOLDER_NAMES[1]:
        output_dir = step_subdir(audio_path.parent.parent, 2)
    else:
        output_dir = audio_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for chunk_number, chunk_audio_path in enumerate(chunk_audio_paths, start=1):
        chunk_srt_path = chunk_audio_path.with_suffix(".srt")
        if chunk_srt_path.is_file() and chunk_srt_path.stat().st_size > 0:
            # A previous run already transcribed this chunk (e.g. a stitch
            # or expected-file check failed afterward) - keep its SRT.
            print(
                f"Chunk {chunk_number:02d}/{len(chunk_audio_paths):02d}: "
                f"{chunk_srt_path.name} already transcribed, reusing it.",
                flush=True,
            )
            chunk_srt_paths.append(chunk_srt_path)
            continue
        print(
            f"Running Whisper for chunk "
            f"{chunk_number:02d}/{len(chunk_audio_paths):02d}: {chunk_audio_path.name}",
            flush=True,
        )
        _transcribe_audio_chunk(chunk_audio_path, chunk_srt_path)
        chunk_srt_paths.append(chunk_srt_path)

    print("All chunk SRTs finished; stitching connected timestamps.", flush=True)
    output_path = output_dir / audio_path.with_suffix(".srt").name
    return stitch_transcription_chunks(chunk_srt_paths, chunk_offsets_ms, output_path)


def split_caption_block(block: str, start_index: int) -> list[str]:
    lines = re.split(r"\r?\n", block)
    if len(lines) < 3 or "-->" not in lines[1]:
        lines[0] = str(start_index)
        return ["\r\n".join(lines)]

    start_text, end_text = [part.strip() for part in lines[1].split("-->", 1)]
    start_ms = parse_srt_time(start_text)
    end_ms = parse_srt_time(end_text)
    subtitle_text = normalize_caption_text(lines[2:])
    if not subtitle_text:
        return []

    return [make_srt_block(start_index, start_ms, end_ms, subtitle_text)]

def fix_srt(input_path: Path, out_dir: Path | None = None) -> Path:
    """Fix bad Whisper SRT timestamps, collapse adjacent repeats, and renumber."""
    input_path = input_path.resolve()
    output_path = (out_dir or input_path.parent) / f"{input_path.stem}_fixed.srt"

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

    write_text_crlf(output_path, "\r\n\r\n".join(renumbered_blocks), encoding="utf-8")
    print(f"Fixed file saved as: {output_path}")
    return output_path

def merge_entries_for_analysis(entries: list[SubtitleEntry]) -> list[tuple[int, str]]:
    """Regroups the on-screen captions into fuller, more natural chunks for
    the LLM discovery passes to read. Each merged chunk keeps the timestamp
    of whichever caption started it, which stays a precise, real anchor for
    the anti-hallucination timestamp check downstream in
    analyze_highlights_emotion.py.

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

def split_srt_into_chunks(input_path: Path, chunk_minutes: int = DEFAULT_CHUNK_MINUTES, out_dir: Path | None = None) -> list[Path]:
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
        part_path = (out_dir or input_path.resolve().parent) / f"transcript_part{index}.txt"
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
    """Generate Step 1 for a local VOD with a separate mic track.

    Extraction, chunk splitting, and chunk persistence all finish here. Step 2
    only launches Whisper against the saved chunk files.
    """
    mic_wav_name = f"{base_name}_mic.wav"
    video_path = target_folder / f"{base_name}{video_suffix}"
    wav_path = step_subdir(target_folder, 1) / mic_wav_name
    organizer_script = SCRIPT_DIR / "OrganizeVODAndFixSRT_Emotion.py"
    audio_filters = (
        f"agate=threshold={NOISE_GATE_THRESHOLD_DB}dB:"
        f"ratio={NOISE_GATE_RATIO}:"
        f"attack={NOISE_GATE_ATTACK_MS}:"
        f"release={NOISE_GATE_RELEASE_MS}"
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
echo Step 1a complete: mic audio extracted to {mic_wav_name}
echo Step 1b/1c: splitting and saving all mic audio chunks...
python -u "{batch_quote(organizer_script)}" --prepare-audio-chunks "{batch_quote(wav_path)}" --no-pause
if errorlevel 1 (
    echo.
    echo ERROR: failed to split and save the mic audio chunks.
    if not "%RUN_ALL%"=="1" pause
    exit /b 1
)
echo.
echo Done! Mic audio and saved chunks are ready for 2_TranscribeAudio.bat.
if not "%RUN_ALL%"=="1" pause
exit /b 0
'''


def make_extract_mic_bat_singletrack(target_folder: Path, base_name: str, video_suffix: str,
                                      isolate_script: Path) -> str:
    """Generate Step 1 for a merged-track VOD.

    Step 1 explicitly completes the separation pipeline before transcription:
    extract the full mix, persist Demucs input chunks, persist trimmed vocal
    chunks, concatenate and render the isolated mic track, then persist the
    separate Whisper chunks.
    """
    mic_wav_name = f"{base_name}_mic.wav"
    mixed_wav_name = f"{base_name}_mixed_full.w64"
    # _mic_demucs_combined.w64: the Demucs-separated (vocals-only) chunks
    # concatenated, pre-render. The final *_mic.wav is the later ffmpeg
    # re-render (16 kHz mono + noise gate) Whisper consumes - the names must
    # not suggest they are interchangeable.
    combined_wav_name = f"{base_name}_mic_demucs_combined.w64"
    demucs_chunk_dir_name = f"{base_name}_demucs_chunks"
    mic_chunk_dir_name = f"{base_name}_mic_demucs_chunks"
    video_path = target_folder / f"{base_name}{video_suffix}"
    step1_dir = step_subdir(target_folder, 1)
    mixed_wav_path = step1_dir / mixed_wav_name
    mic_wav_path = step1_dir / mic_wav_name
    combined_wav_path = step1_dir / combined_wav_name
    demucs_chunk_dir = step1_dir / demucs_chunk_dir_name
    mic_chunk_dir = step1_dir / mic_chunk_dir_name
    organizer_script = SCRIPT_DIR / "OrganizeVODAndFixSRT_Emotion.py"
    return f'''@echo off
echo This VOD has one merged audio track (Twitch-style).
echo Step 1 will extract the full mix, split it for Demucs, separate each
echo chunk, save the separated mic chunks, combine and render the mic track,
echo then split the rendered track for Whisper.
echo The full mix remains available as {mixed_wav_name}.
echo.
if exist "{batch_quote(mixed_wav_path)}" (
    echo Found existing extracted full mix: {mixed_wav_name}
    echo Reusing it and skipping full-mix extraction.
    goto isolate_vocals
)
echo Step 1a: extracting full mix from {base_name}{video_suffix} ...
ffmpeg -y -i "{batch_quote(video_path)}" -map 0:a:0 -ar 44100 -ac 2 -f w64 "{batch_quote(mixed_wav_path)}"
if errorlevel 1 (
    echo.
    echo ERROR: ffmpeg failed to extract the mixed audio track. Make sure
    echo ffmpeg is installed and on your PATH, and that the video is in this
    echo folder.
    if not "%RUN_ALL%"=="1" pause
    exit /b 1
)

:isolate_vocals
echo.
echo Step 1b/1c: splitting the full mix for Demucs and separating each chunk...
python -u "{batch_quote(isolate_script)}" "{batch_quote(mixed_wav_path)}" "{batch_quote(mic_wav_path)}" ^
    --demucs-chunk-dir "{batch_quote(demucs_chunk_dir)}" ^
    --mic-chunk-dir "{batch_quote(mic_chunk_dir)}" ^
    --combined-path "{batch_quote(combined_wav_path)}"
if errorlevel 1 (
    echo.
    echo ERROR: vocal isolation failed. See the output above - common causes
    echo are the demucs package not being installed ^(re-run
    echo Install_PogEngine.bat^) or running out of GPU memory on a long
    echo VOD ^(try setting the VOCAL_ISOLATION_SEGMENT_SECONDS environment
    echo variable - see pipeline_config.py^).
    if not "%RUN_ALL%"=="1" pause
    exit /b 1
)
echo.
echo Step 1d: mic chunks saved in {mic_chunk_dir_name}
echo Step 1e: Demucs-separated mic audio combined into {combined_wav_name}
echo Step 1f: rendering the isolated mic track to {mic_wav_name} is complete.
echo Step 1g: splitting the rendered mic track into Whisper chunks...
python -u "{batch_quote(organizer_script)}" --prepare-audio-chunks "{batch_quote(mic_wav_path)}" --no-pause
if errorlevel 1 (
    echo.
    echo ERROR: failed to split and save the rendered mic audio chunks.
    if not "%RUN_ALL%"=="1" pause
    exit /b 1
)
echo.
echo Done! Rendered mic audio and saved Whisper chunks are ready for 2_TranscribeAudio.bat.
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

def make_transcribe_bat(script_path: Path) -> str:
    return f'''@echo off
if "%~1"=="" (
    echo Drag your prepared _mic.wav file onto this script to run Whisper.
    echo Step 1 must finish first so all audio chunks already exist.
    if not "%RUN_ALL%"=="1" pause
    exit /b 1
)

set AUDIO=%~1

echo Step 2a: running Whisper on each saved audio chunk...
echo Whisper model and VAD settings are loaded from the organizer script.
echo.
python -u "{batch_quote(script_path)}" --transcribe-audio "%AUDIO%" --no-pause

if errorlevel 1 (
    echo.
    echo ERROR: Whisper chunk transcription failed.
    echo Saved chunk audio and completed chunk SRTs remain in the chunk folder.
    if not "%RUN_ALL%"=="1" pause
    exit /b 1
)

echo.
echo Step 2b complete: all chunk SRTs stitched into %~n1.srt
echo Next: drag the .srt file onto 3_FixSRT.bat
if not "%RUN_ALL%"=="1" pause
exit /b 0
'''

def make_llama_server_bat() -> str:
    """Launcher for llama.cpp llama-server (llamacpp backend). Generated on every
    organize_video() so the VOD folder always has the current config snapshot.
    For the managed two-model mode the pipeline itself starts/stops the server
    and hot-swaps GGUFs (discovery -> judge) - this bat is only for manual
    testing. Harmless when LLM_BACKEND is ollama."""
    # Parse host/port from LLAMA_SERVER_URL (default http://localhost:8080).
    url = LLAMA_SERVER_URL.strip()
    host = "127.0.0.1"
    port = "8080"
    try:
        without_scheme = url.split("://", 1)[-1]
        host_port = without_scheme.split("/", 1)[0]
        if ":" in host_port:
            h, p = host_port.rsplit(":", 1)
            if h:
                host = "127.0.0.1" if h in ("localhost", "0.0.0.0", "") else h
            if p.isdigit():
                port = p
        elif host_port:
            host = "127.0.0.1" if host_port in ("localhost", "0.0.0.0", "") else host_port
    except Exception:
        pass
    def _resolve(role: str) -> str:
        if role == "discovery":
            p = (LLAMA_DISCOVERY_MODEL_PATH or "").strip()
            if p:
                return p
            return (LLAMA_MODEL_PATH or "").strip()
        p = (LLAMA_JUDGE_MODEL_PATH or "").strip()
        if p:
            return p
        p2 = (LLAMA_DISCOVERY_MODEL_PATH or "").strip()
        if p2:
            return p2
        return (LLAMA_MODEL_PATH or "").strip()
    disc_path = _resolve("discovery")
    judge_path = _resolve("judge")
    ctx = int(LLAMA_CONTEXT_SIZE) if str(LLAMA_CONTEXT_SIZE).strip().isdigit() else 8192
    # Display strings
    disc_display = disc_path if disc_path else "<set LLAMA_DISCOVERY_MODEL_PATH>"
    judge_display = judge_path if judge_path else "<set LLAMA_JUDGE_MODEL_PATH>"
    # Launch line for manual mode - use discovery GGUF (first stage). Judge GGUF shown as comment.
    if disc_path:
        disc_arg = f'"{batch_quote(Path(disc_path))}"'
        launch_line = f'llama-server --model {disc_arg} -c {ctx} --host {host} --port {port} --n-gpu-layers 99'
        launch_comment = f":: Discovery GGUF: {disc_display}"
        if judge_path and judge_path != disc_path:
            judge_arg = f'"{batch_quote(Path(judge_path))}"'
            launch_comment += f"\n:: Judge GGUF:      {judge_display}\n:: For judge manually: llama-server --model {judge_arg} -c {ctx} --host {host} --port {port} --n-gpu-layers 99"
    else:
        launch_line = f':: Set LLAMA_*_MODEL_PATH first, then edit this line:\n:: llama-server --model "C:\\path\\to\\model.gguf" -c {ctx} --host {host} --port {port} --n-gpu-layers 99'
        launch_comment = f":: Discovery: {disc_display}\n:: Judge:     {judge_display}"
    return f'''@echo off
echo llama-server launcher for Pog Engine (llamacpp backend - managed hot-swap).
echo Config snapshot at generation time:
echo   LLM_BACKEND={LLM_BACKEND}
echo   Discovery GGUF: {disc_display}
echo   Judge GGUF:     {judge_display}
echo   Context (-c): {ctx}
echo   URL: {url}  (--host {host} --port {port})
echo.
if "{LLM_BACKEND}"=="llamacpp" (
    echo This backend is active. The pipeline now MANAGES llama-server itself:
    echo  - Starts with discovery GGUF for discovery
    echo  - Stops to free VRAM for emotion
    echo  - Restarts with judge GGUF for audioscan/verify/judge
    echo This launcher is OPTIONAL - only for manual testing.
    echo For manual: it starts the discovery GGUF below. Edit for judge if needed.
) else (
    echo NOTE: LLM_BACKEND is currently "ollama", so this launcher is not needed
    echo for this VOD. Switch to llamacpp in the configurator to use it.
)
echo.
{launch_comment}
{launch_line}
if errorlevel 1 (
    echo.
    echo ERROR: llama-server failed to start. Common causes:
    echo  - llama-server.exe not on PATH
    echo  - GGUF path wrong or file missing
    echo  - port {port} already in use
    pause
    exit /b 1
)
'''

def _make_srt_step_bat(
    script_path: Path,
    *,
    missing_message: str,
    progress_message: str,
    command: str,
    error_label: str,
    done_message: str,
    next_message: str,
    out_dir: Path | None = None,
) -> str:
    out_dir_line = f" --out-dir \"{batch_quote(str(out_dir))}\"" if out_dir is not None else ""
    return f'''@echo off
if "%~1"=="" (
    echo {missing_message}
    if not "%RUN_ALL%"=="1" pause
    exit /b 1
)

set SRT=%~1

echo {progress_message}: %~nx1
echo.
python "{batch_quote(script_path)}" {command} "%SRT%"{out_dir_line} --no-pause

if errorlevel 1 (
    echo.
    echo ERROR: {error_label} failed.
    if not "%RUN_ALL%"=="1" pause
    exit /b 1
)

echo.
echo {done_message}
echo {next_message}
if not "%RUN_ALL%"=="1" pause
exit /b 0
'''


def make_fix_srt_bat(script_path: Path, target_folder: Path) -> str:
    return _make_srt_step_bat(
        script_path,
        missing_message="Drag your .srt file onto this script.",
        progress_message="Fixing SRT timestamps and adjacent repeats",
        command="--fix-srt",
        error_label="SRT fix",
        done_message="Done! Fixed SRT saved in the VOD folder as *_fixed.srt.",
        next_message="Next: drag the *_fixed.srt file onto 4_SplitSRT.bat",
        out_dir=target_folder,
    )


def make_split_srt_bat(script_path: Path, target_folder: Path) -> str:
    return _make_srt_step_bat(
        script_path,
        missing_message="Drag your fixed .srt file onto this script.",
        progress_message="Splitting SRT into transcript_part files",
        command="--split-srt",
        error_label="SRT split",
        done_message="Done! transcript_part files are saved in the step4_split_srt folder.",
        next_message="Double-click 5_AnalyzeHighlights.bat or Run_Pog_Engine.bat to find highlights.",
        out_dir=step_subdir(target_folder, 4),
    )

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

def make_debug_stage_bat(target_folder: Path, stage_key: str, step_label: str) -> str:
    """5a-5f: force-reruns exactly one internal stage, for debugging (e.g.
    after tweaking a prompt). NOT part of the main 1-6 sequence and not
    tracked by the RunAll GUI - double-click these directly when you want
    to. Forcing a stage clears every checkpoint after it, since they'd
    otherwise be stale leftovers from before the change."""
    vod_root = batch_quote(str(target_folder))
    return f'''@echo off
echo Force-running {step_label} on "{vod_root}" (debug - clears later checkpoints)...
echo.
python -u "{batch_quote(ANALYZE_HIGHLIGHTS)}" "{vod_root}" --stage {stage_key}
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
    """Launch the Tk GUI without keeping a console window open.

    ``start /b`` lets the batch file exit immediately while ``pythonw`` keeps
    the GUI process detached from the batch file's console.
    """
    return f'''@echo off
start "" /b pythonw.exe "{batch_quote(script_path)}" --run-all-gui "{batch_quote(target_folder)}" --base-name "{batch_quote(base_name)}" --no-pause
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
        # Step 1 is complete only when the mic WAV and every saved chunk exist.
        RunAllStep(BIG_STEP_LABELS[0], "1_ExtractMicAudio.bat", expected_kind="prepared_chunks"),
        RunAllStep(BIG_STEP_LABELS[1], "2_TranscribeAudio.bat", "mic_wav", True, "raw_srt"),
        RunAllStep(BIG_STEP_LABELS[2], "3_FixSRT.bat", "raw_srt", True, "fixed_srt"),
        RunAllStep(BIG_STEP_LABELS[3], "4_SplitSRT.bat", "fixed_srt", True, "transcript_part"),
        # One external process from the GUI's view, but internally walks 6
        # checkpointed sub-stages (discovery -> ... -> export) with its own
        # resume logic - see analyze_highlights_emotion.py. "Done" here means
        # the final CSV exists; the script decides what still needs to run.
        RunAllStep(BIG_STEP_LABELS[4], "5_AnalyzeHighlights.bat", "transcript_part", False, "highlights_csv"),
    ]




ANALYSIS_STAGE_DETAILS = {
    "5a. Discovery": {
        "task": "Discovery: transcript candidate passes",
        "model": f"Model: {MODEL}",
        "next": "Next: Audio Scan — model-free energy analysis",
    },
    "5b. Audio Scan": {
        "task": "Audio Scan: model-free energy analysis",
        "model": f"Title model when needed: {JUDGE_MODEL}",
        "next": "Next: Emotion Scoring — load speech-emotion model",
    },
    "5c. Emotion Scoring": {
        "task": "Emotion Scoring: loading speech-emotion model",
        "model": f"Model: {EMOTION_MODEL_ID}" if EMOTION_ENABLED else "Model: disabled by configuration",
        "next": "Next: Verification — check candidates against transcript",
    },
    "5d. Verification": {
        "task": "Verification: checking candidates against transcript",
        "model": f"Model: {JUDGE_MODEL}",
        "next": "Next: Judging — rank the candidate pool",
    },
    "5e. Judging": {
        "task": "Judging: ranking the candidate pool",
        "model": f"Model: {JUDGE_MODEL}",
        "next": "Next: Export — write CSV, EDL, and run metadata",
    },
    "5f. Export": {
        "task": "Export: writing CSV, EDL, and run metadata",
        "model": "Model: none",
        "next": "",
    },
}
BIG_STEP_MINI_STAGES = {
    0: (
        ("1a", "Extract mic audio"),
        ("1b", "Split mic audio into chunks"),
        ("1c", "Save chunks"),
    ),
    1: (
        ("2a", "Run Whisper on each chunk"),
        ("2b", "Stitch the chunk SRTs"),
    ),
    2: (
        ("3a", "Read captions"),
        ("3b", "Repair timestamps"),
        ("3c", "Remove repeats"),
        ("3d", "Write fixed SRT"),
    ),
    3: (
        ("4a", "Read fixed captions"),
        ("4b", "Merge into thoughts"),
        ("4c", "Assign time chunks"),
        ("4d", "Write transcript parts"),
    ),
    4: (
        ("5a", "Discovery"),
        ("5b", "Audio Scan"),
        ("5c", "Emotion"),
        ("5d", "Verify"),
        ("5e", "Judge"),
        ("5f", "Export"),
    ),
}
SINGLE_TRACK_STEP_MINI_STAGES = (
    ("1a", "Extract full mixed audio"),
    ("1b", "Split Demucs input chunks"),
    ("1c", "Run Demucs per chunk"),
    ("1d", "Save separated mic chunks"),
    ("1e", "Combine separated mic audio"),
    ("1f", "Render 16 kHz mono mic"),
    ("1g", "Prepare Whisper chunks"),
)


def _mini_stages_for_step(
    index: int,
    target_folder: Path | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return the visible mini-processes for one numbered pipeline step."""
    if index == 0 and _vod_is_single_track(target_folder) is True:
        return SINGLE_TRACK_STEP_MINI_STAGES
    return BIG_STEP_MINI_STAGES[index]


ANALYSIS_STAGE_BY_LABEL = {
    label.casefold(): (label.split(".", 1)[0], label.split(". ", 1)[1])
    for label in ANALYSIS_STAGE_DETAILS
}


def _vod_is_single_track(target_folder: Path | None) -> bool | None:
    if target_folder is None:
        return None
    try:
        metadata = json.loads((target_folder / "vod_audio_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    value = metadata.get("single_track_mode")
    return value if isinstance(value, bool) else None


def _step_model_details(index: int, target_folder: Path | None = None) -> tuple[str, str]:
    if index == 0:
        if _vod_is_single_track(target_folder) is False:
            return "Process: extract separate mic track, then split and save Whisper chunks", ""
        if _vod_is_single_track(target_folder) is True:
            return (
                "Process: extract mix → Demucs chunks → mic chunks → render → Whisper chunks",
                f"Vocal model: {VOCAL_ISOLATION_MODEL}",
            )
        return (
            "Process: extract/isolate mic audio, then split and save Whisper chunks",
            f"Vocal model when needed: {VOCAL_ISOLATION_MODEL}",
        )
    if index == 1:
        return (
            "Process: Whisper each saved chunk, then stitch shifted timestamps",
            f"Whisper model: {Path(WHISPER_MODEL).name}",
        )
    if index == 2:
        return "Process: repair timestamps and remove adjacent repeats", ""
    if index == 3:
        return "Process: regroup captions into transcript_part files", ""
    return (
        "Process: discovery → audio scan → emotion → verification → judging → export",
        f"Models: {MODEL} / {JUDGE_MODEL} / {EMOTION_MODEL_ID if EMOTION_ENABLED else 'emotion disabled'}",
    )


def _initial_next_detail(index: int, target_folder: Path | None = None) -> str:
    if index == 0:
        if _vod_is_single_track(target_folder) is False:
            return "Next: Extract, then split and save Whisper chunks"
        if _vod_is_single_track(target_folder) is True:
            return "Next: Extract mix, separate Demucs chunks, render, then prepare Whisper chunks"
        return "Next: Extract/isolate mic audio, then split and save Whisper chunks"
    if index == 1:
        return "Next: Run Whisper on each saved chunk"
    if index == 2:
        return "Next: Repair timestamps and remove repeats"
    if index == 3:
        return "Next: Regroup captions into transcript parts"
    return "Next: Discovery — transcript candidate passes"


def _strip_console_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-_]", "", text).strip()


def _console_log_tag(text: str) -> str:
    lowered = text.casefold()
    if "skipping because output" in lowered or "checkpoint already exists, skipping" in lowered:
        return "success"
    if (
        "[warn" in lowered
        or "[!]" in lowered
        or "retrying" in lowered
        or "continuing without" in lowered
        or "skipping" in lowered
    ):
        return "warning"
    if (
        "error" in lowered
        or "failed" in lowered
        or "exception" in lowered
        or re.search(r"exit code \d+", lowered)
    ):
        return "failure"
    if (
        "[ok]" in lowered
        or "done" in lowered
        or "saved" in lowered
        or "finished" in lowered
        or "success" in lowered
        or "found output" in lowered
        or "complete" in lowered
    ):
        return "success"
    return "info"

def newest_file(paths: Iterable[Path]) -> Path | None:
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return None
    return max(existing_paths, key=lambda path: path.stat().st_mtime)

def run_all_file_info(target_folder: Path, base_name: str, kind: str) -> tuple[Path | None, str]:
    """Locate a step's input/output file. New-layout artifacts live in the
    step's own folder; legacy VOD folders kept them at the root, so both are
    checked (step folder wins)."""
    step_dirs = {n: step_subdir(target_folder, n) for n in (1, 2, 3, 4, 5)}

    def newest_in(bases, pattern):
        """Newest match inside the given folders (step folder wins)."""
        found = []
        for base in bases:
            if base.is_dir():
                found.extend(base.glob(pattern))
        return newest_file(found) if found else None

    if kind == "mic_wav":
        exact_path = step_dirs[1] / f"{base_name}_mic.wav"
        candidates = newest_in((step_dirs[1], target_folder), "*_mic.wav")
        description = f"{exact_path} or newest *_mic.wav"
    elif kind == "prepared_chunks":
        # the chunk set lives in the step 1 folder; legacy folders kept it at
        # the root - validate whichever exists (mic wav + manifest present)
        for chunks_dir, mic_dir in (
            (step_dirs[1] / f"{base_name}_mic_transcription_chunks", step_dirs[1]),
            (target_folder / f"{base_name}_mic_transcription_chunks", target_folder),
        ):
            mic_path = mic_dir / f"{base_name}_mic.wav"
            if not (mic_path.is_file() and (chunks_dir / "chunk_manifest.json").is_file()):
                continue
            try:
                _load_prepared_audio_chunks(mic_path)
            except (OSError, RuntimeError, ValueError, TypeError):
                continue
            return chunks_dir, f"{mic_path} plus saved chunks in {chunks_dir}"
        return None, "prepared chunks not found (run step 1 first)"
    elif kind == "raw_srt":
        exact_path = step_dirs[2] / f"{base_name}_mic.srt"
        raw_candidates = [
            path
            for base in (step_dirs[2], target_folder) if base.is_dir()
            for path in base.glob("*.srt")
            if not path.stem.casefold().endswith("_fixed")
        ]
        candidates = newest_file(raw_candidates) if raw_candidates else None
        description = f"{exact_path} or newest non-fixed *.srt"
    elif kind == "fixed_srt":
        exact_path = target_folder / f"{base_name}_mic_fixed.srt"
        raw_candidates = list(target_folder.glob("*_fixed.srt"))
        candidates = newest_file(raw_candidates) if raw_candidates else None
        description = f"{exact_path} or newest *_fixed.srt"
    elif kind == "transcript_part":
        exact_path = step_dirs[4] / "transcript_part1.txt"
        candidates = newest_in((step_dirs[4], target_folder), "transcript_part*.txt")
        description = f"{exact_path} or newest transcript_part*.txt"
    elif kind == "highlights_csv":
        exact_path = None
        candidates = newest_in((step_dirs[5], target_folder), "top*_highlights.csv")
        description = f"{step_dirs[5]} / top*_highlights.csv"
    else:
        raise ValueError(f"Unknown run-all file kind: {kind}")

    path = exact_path if exact_path is not None and exact_path.exists() else candidates
    return path, description

def gallery_image_paths(gallery_dir: Path = GALLERY_DIR) -> list[Path]:
    if not gallery_dir.exists():
        return []

    image_paths = [
        path
        for path in gallery_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in GALLERY_IMAGE_EXTENSIONS
    ]
    return sorted(image_paths, key=lambda path: path.stat().st_mtime, reverse=True)


def record_big_step_duration(
    *,
    run_id: str,
    target_folder: Path,
    step_number: int,
    step_label: str,
    status: str,
    duration_seconds: float,
    started_at: datetime,
    finished_at: datetime,
) -> bool:
    """Append one RunAll big-step timing row to the project history CSV."""
    history_path = SCRIPT_DIR / STEP_HISTORY_FILENAME
    row = {
        "run_timestamp": finished_at.isoformat(timespec="seconds"),
        "run_id": run_id,
        "stream_folder": str(target_folder),
        "step_number": step_number,
        "step_label": step_label,
        "status": status,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_seconds": round(max(duration_seconds, 0.0), 1),
    }
    fieldnames = list(row)
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        is_new_file = not history_path.exists() or history_path.stat().st_size == 0
        with history_path.open("a", newline="", encoding="utf-8") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=fieldnames)
            if is_new_file:
                writer.writeheader()
            writer.writerow(row)
    except OSError as exc:
        print(f"[!] Could not write big-step duration history: {exc}")
        return False
    return True

def latest_completed_step_duration(target_folder: Path, step_number: int) -> float | None:
    """Return the last successful duration for this VOD and big step."""
    history_path = SCRIPT_DIR / STEP_HISTORY_FILENAME
    if not history_path.is_file():
        return None

    target_key = os.path.normcase(os.path.abspath(str(target_folder)))
    latest_duration: float | None = None
    try:
        with history_path.open("r", newline="", encoding="utf-8") as history_file:
            for row in csv.DictReader(history_file):
                if row.get("status") != "Done":
                    continue
                try:
                    row_step = int(row.get("step_number", ""))
                    duration = float(row.get("duration_seconds", ""))
                except (TypeError, ValueError):
                    continue
                row_folder = row.get("stream_folder", "")
                if row_step == step_number and os.path.normcase(os.path.abspath(row_folder)) == target_key:
                    latest_duration = max(duration, 0.0)
    except OSError:
        return None
    return latest_duration


def make_step6_log_path(target_folder: Path) -> Path:
    return target_folder / f"step6_run_{datetime.now():%Y%m%d_%H%M%S}.log"

def _last_input_tick() -> int | None:
    """Return the Windows tick count (ms) of the last mouse/keyboard input.

    Backs the RunAll GUI's auto-sleep countdown: a changed tick during the
    wait means the user touched the machine and the pending sleep must be
    cancelled. Returns None when the query fails (non-Windows, API error),
    which callers treat as "cannot detect input".
    """
    try:
        import ctypes
        from ctypes import wintypes

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.UINT),
                ("dwTime", wintypes.DWORD),
            ]

        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return None
        return int(info.dwTime)
    except Exception:
        return None

RUN_ALL_VERSION = "V2.0.0"  # shown in the RunAll GUI header (per GUI REDESIGN CONCEPT mockups)

import tkinter as tk  # noqa: E402  (RunAll GUI helpers below live at module level)

# ---- RunAll GUI design tokens (GUI REDESIGN CONCEPT / Color and Font Guidelines) ----
GUI_BG = "#181818"            # black background
GUI_WHITE = "#f9f9f9"         # white element (borders, body text)
GUI_ORANGE = "#ff8b1b"        # any orange element (labels, wires, accents)
GUI_TEXT_DARK = "#181818"     # black text on colored cells
GUI_NEON_GREEN = "#00ff7e"    # completed step
GUI_PROGRESS_GREEN = "#1d874d"  # step in progress
GUI_PURE_RED = "#ff0000"      # error / step failed
GUI_PROGRESS_RED = "#751313"  # waiting for previous step

GUI_STATE_COLORS = {
    "Waiting": GUI_PROGRESS_RED,
    "Running": GUI_PROGRESS_GREEN,
    "Done": GUI_NEON_GREEN,
    "Skipped": GUI_NEON_GREEN,
    "Failed": GUI_PURE_RED,
    "Stopped": GUI_ORANGE,  # extension: stopped is neither failed nor waiting
}

# What each mini-process actually does, for the hover tooltip. Keyed by big-step
# index then mini code; the single-track (Twitch) route swaps step 1's set.
MINI_DESCRIPTIONS = {
    0: {
        "1a": "ffmpeg maps the separate mic track (audio stream 1) out of the VOD into *_mic.wav.",
        "1b": "The mic WAV is cut into overlapping transcription windows (30 min + 10 s overlap) so every Whisper window starts with a fresh decoder context.",
        "1c": "Every chunk WAV plus chunk_manifest.json is saved under *_mic_transcription_chunks/ and reused on reruns.",
    },
    1: {
        "2a": "A fresh whisper-cli process decodes each saved chunk (GPU-accelerated on NVIDIA; CPU-bound on AMD, where whisper.cpp ships no GPU Windows build); overlap captions are deduplicated after every chunk succeeds.",
        "2b": "Chunk SRTs shift into the full-audio clock, drop exact overlap duplicates, and stitch into the raw SRT.",
    },
    2: {
        "3a": "The raw Whisper SRT is parsed into caption blocks.",
        "3b": "Out-of-order or bad Whisper timestamps are repaired.",
        "3c": "Adjacent duplicate sentences and blocks collapse into a single caption.",
        "3d": "Captions are renumbered and written to *_mic_fixed.srt (one retained caption per block; display wrapping only).",
    },
    3: {
        "4a": "The fixed SRT is loaded as caption entries.",
        "4b": "Captions regroup into fuller thoughts (~30 words) for analysis.",
        "4c": "Thoughts are grouped into ~25-minute analysis chunks.",
        "4d": "transcript_partN.txt files are written - the highlight analyzer's input.",
    },
    4: {
        "5a": "Multiple LLM passes over each transcript part propose highlight candidates (Emotion / Gameplay / Viral prompts); junk titles and hallucinated timestamps are filtered.",
        "5b": "Model-free DSP pass over the mic track: loudness + speech-rate arousal peaks not near an LLM candidate become new candidates (catches wordless reactions).",
        "5c": "The local speech-emotion model scores audio around each candidate; emotion and hype-phrase boosts are applied to the scores.",
        "5d": "The judge LLM checks each candidate batch against the transcript; unsupported or hallucinated timestamps are dropped.",
        "5e": "A tournament batch ranks the candidate pool down to TOP_N with a 5-factor priority order.",
        "5f": "Rank-calibrated scores become top<N>_highlights.csv, a Resolve EDL marker file, and run_info.json.",
    },
}
MINI_DESCRIPTIONS_SINGLE_TRACK_STEP1 = {
    "1a": "ffmpeg extracts the full mixed track into a Wave64 *_mixed_full.w64 (64-bit size field - no 4 GB WAV ceiling on long VODs).",
    "1b": "The full mix is persisted as 10-minute Demucs input chunks.",
    "1c": "Demucs separates each input chunk - one run per chunk, GPU when available.",
    "1d": "Trimmed separated mic chunks are saved under *_mic_demucs_chunks/.",
    "1e": "Separated mic chunks are concatenated into *_mic_demucs_combined.w64.",
    "1f": "The combined vocal audio renders to *_mic.wav at 16 kHz mono with the shared noise gate.",
    "1g": "The rendered mic WAV is cut into overlapping Whisper windows with a manifest.",
}


def mini_description(index: int, code: str, target_folder: Path | None = None) -> str:
    """Tooltip body text for one mini-process cell."""
    if index == 0 and _vod_is_single_track(target_folder) is True:
        return MINI_DESCRIPTIONS_SINGLE_TRACK_STEP1.get(code, "")
    return MINI_DESCRIPTIONS.get(index, {}).get(code, "")


def _resolve_font_family(root: object, preferred: list[str]) -> str:
    """First installed font family from `preferred` (case-insensitive).

    The concept pins Helvetica Compressed / Helvetica Regular; Windows installs
    neither by default, so this walks a metric-compatible fallback chain and
    only lands on the last name when nothing better exists.
    """
    import tkinter.font as tkfont

    try:
        available = {name.casefold(): name for name in tkfont.families(root)}
    except Exception:
        return preferred[-1]
    for candidate in preferred:
        if candidate.casefold() in available:
            return available[candidate.casefold()]
    return preferred[-1]


def _blend_hex(from_color: str, to_color: str, t: float) -> str:
    """Linear RGB blend between two #rrggbb colors (t in 0..1) for fade-ins."""
    def channel(text: str, shift: int) -> int:
        return int(text[shift:shift + 2], 16)

    t = max(0.0, min(1.0, t))
    r = round(channel(from_color, 1) + (channel(to_color, 1) - channel(from_color, 1)) * t)
    g = round(channel(from_color, 3) + (channel(to_color, 3) - channel(from_color, 3)) * t)
    b = round(channel(from_color, 5) + (channel(to_color, 5) - channel(from_color, 5)) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _make_titled_panel(
    parent: object,
    title: str | None,
    *,
    bg: str,
    border: str,
    title_fg: str,
    title_font: tuple,
    border_px: int = 2,
) -> tuple[object, object, object | None]:
    """Black panel with a crisp white outline and the orange title sitting ON
    the border line (mockup style). Returns (wrap_frame, inner_frame, title_label)."""
    top_pad = 9 if title else 0
    wrap = tk.Frame(parent, bg=bg)
    inner = tk.Frame(
        wrap,
        bg=bg,
        highlightthickness=border_px,
        highlightbackground=border,
        highlightcolor=border,
    )
    inner.pack(fill="both", expand=True, pady=(top_pad, 0))
    title_label = None
    if title:
        title_label = tk.Label(
            wrap,
            text=f" {title} ",
            bg=bg,
            fg=title_fg,
            font=title_font,
            bd=0,
        )
        title_label.place(x=14, y=top_pad, anchor="w")
    return wrap, inner, title_label


class _GhostButton(tk.Frame):
    """Ghost button with a crisp state-colored outline and hover feedback.

    The outline is drawn by this wrapper frame (tk.Button's own highlight ring
    renders faintly on Windows), and <Enter>/<Leave> fills the button orange
    with black text - Tk buttons have no hover state natively."""

    def __init__(self, parent, text, command=None, *, font, padx=14, pady=5, ring=1):
        super().__init__(
            parent, bg=GUI_BG,
            highlightthickness=ring,
            highlightbackground=GUI_ORANGE, highlightcolor=GUI_ORANGE,
        )
        self._button = tk.Button(
            self,
            text=text,
            command=command,
            bg=GUI_BG,
            fg=GUI_ORANGE,
            activebackground=GUI_ORANGE,
            activeforeground=GUI_TEXT_DARK,
            disabledforeground=GUI_PROGRESS_RED,
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=padx,
            pady=pady,
            cursor="hand2",
            font=font,
        )
        self._button.pack(fill="both", expand=True)
        self._button.bind("<Enter>", self._on_enter)
        self._button.bind("<Leave>", self._on_leave)

    @property
    def button(self) -> tk.Button:
        return self._button

    def _on_enter(self, _event: object = None) -> None:
        if str(self._button["state"]) != "disabled":
            self._button.configure(bg=GUI_ORANGE, fg=GUI_TEXT_DARK)

    def _on_leave(self, _event: object = None) -> None:
        if str(self._button["state"]) != "disabled":
            self._button.configure(bg=GUI_BG, fg=GUI_ORANGE)

    def configure(self, **kwargs) -> None:
        ring_color = kwargs.pop("ring_color", None)
        self._button.configure(**kwargs)
        if ring_color is not None:
            super().configure(highlightbackground=ring_color, highlightcolor=ring_color)


def migrate_legacy_vod_folder(target_folder: Path, events: "queue.Queue | None" = None) -> None:
    """One-time layout migration for VOD folders created by older builds.

    Moves each step's artifacts from the folder root into its step folder.
    The big run log (step6_run_*.log), the fixed SRT, and the marker EDL stay
    at the root, as do the runner and step bats. Also renames
    6_RunAllSteps.bat to Run_Pog_Engine.bat for pre-redesign folders."""
    moves = {
        1: ("*_mic.wav", "*_mixed_full.w64", "*_mic_demucs_combined.w64",
            "*_mic_combined.w64",  # pre-rename builds wrote this name
            "*_demucs_chunks", "*_mic_demucs_chunks"),
        2: ("*_mic.srt", "*_mic_transcription_chunks"),
        4: ("transcript_part*.txt",),
        5: ("checkpoint_*.json", "log_discovery.txt", "log_audioscan.txt",
            "log_emotion.txt", "log_verify.txt", "log_judge.txt", "log_export.txt",
            "emotion_scores.csv", "top*_highlights.csv", "run_info.json",
            "pipeline_stats.json", "5a_Discovery.bat", "5b_AudioScan.bat",
            "5c_EmotionScoring.bat", "5d_Verify.bat", "5e_Judge.bat",
            "5f_Export.bat"),
    }

    def announce(message: str) -> None:
        print(message, flush=True)
        if events is not None:
            events.put(("log", message + "\n"))

    moved = 0
    for step, patterns in moves.items():
        step_dir = step_subdir(target_folder, step)
        for pattern in patterns:
            for path in target_folder.glob(pattern):
                if not path.is_file() and not path.is_dir():
                    continue
                step_dir.mkdir(parents=True, exist_ok=True)
                destination = step_dir / path.name
                if destination.exists():
                    continue
                shutil.move(str(path), str(destination))
                moved += 1
                announce(f"[layout] Moved {path.name} -> {STEP_FOLDER_NAMES[step]}/")

    old_runner = target_folder / "6_RunAllSteps.bat"
    new_runner = target_folder / "Run_Pog_Engine.bat"
    if old_runner.is_file() and not new_runner.exists():
        old_runner.rename(new_runner)
        announce("[layout] Renamed 6_RunAllSteps.bat -> Run_Pog_Engine.bat")
    if moved:
        announce(f"[layout] Migration complete: {moved} item(s) organized into step folders.")


def run_all_gui(target_folder: Path, base_name: str) -> int:
    """Run the five pipeline steps and record each attempt's wall time.

    Successful, skipped, failed, and stopped attempts are appended to
    ``View_Pipeline_Duration_History.csv`` beside this script. The duration viewer
    displays averages from successful attempts.

    Visuals follow the GUI REDESIGN CONCEPT: state-colored cells on black
    (neon done / progress green working / pure red failed / dark red waiting),
    sync bars that light on step handoff, a POG ENGINE FINISH bar that only
    turns green when everything passed, breathing glow on lit cells, hover
    tooltips and outlines, and a gallery that yields to a screen-saver button
    when the window is not maximized.
    """
    target_folder = target_folder.resolve()
    steps = build_run_all_steps(target_folder, base_name)

    import tkinter as tk
    import tkinter.font as tkfont
    from tkinter import messagebox

    try:
        from PIL import Image, ImageDraw, ImageFilter, ImageTk
    except ImportError:
        Image = None
        ImageDraw = None
        ImageFilter = None
        ImageTk = None

    events: queue.Queue[tuple[str, object]] = queue.Queue()
    exit_code = {"value": 0}
    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    run_log_path = make_step6_log_path(target_folder)
    run_log = run_log_path.open("w", encoding="utf-8", buffering=1)
    run_log_closed = {"value": False}
    run_log.write("Step 6 run log\n")
    run_log.write(f"Pog Engine: {RUN_ALL_VERSION}\n")
    run_log.write(f"Started: {datetime.now().isoformat(timespec='seconds')}\n")
    run_log.write(f"Run ID: {run_id}\n")
    run_log.write(f"Folder: {target_folder}\n")
    run_log.write(f"Base name: {base_name}\n\n")

    # Stop-button state: current step's subprocess (so Stop can kill it),
    # whether a stop was requested (so a killed step reports "Stopped" not
    # "Failed"), and the last fully-completed step for the status bar.
    current_process: dict[str, subprocess.Popen | None] = {"popen": None}
    stop_requested = {"value": False}

    # Per-step logs: steps 1-4 each get their own log file inside their step
    # folder, fed from the captured subprocess output while that step runs
    # (the combined step6_run_*.log stays at the VOD root; step 5 writes its
    # own per-sub-stage logs).
    step_log_file: dict = {"handle": None, "index": None}

    def open_step_log(index: int) -> None:
        close_step_log()
        log_path = step_subdir(target_folder, index + 1) / f"log_step{index + 1}.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        step_log_file["handle"] = open(log_path, "a", encoding="utf-8")
        step_log_file["handle"].write(
            f"\n{'=' * 60}\nRun started: {datetime.now().isoformat(timespec='seconds')}\n{'=' * 60}\n"
        )
        step_log_file["index"] = index

    def close_step_log() -> None:
        handle = step_log_file["handle"]
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        step_log_file["handle"] = None
        step_log_file["index"] = None

    def write_step_log(index: int, line: str) -> None:
        handle = step_log_file["handle"]
        if handle is not None and step_log_file["index"] == index:
            try:
                handle.write(line)
                handle.flush()
            except Exception:
                pass

    def write_run_log(text: str) -> None:
        if run_log_closed["value"]:
            return
        run_log.write(text)
        run_log.flush()

    def close_run_log() -> None:
        if run_log_closed["value"]:
            return
        run_log.write(f"\nClosed: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        run_log.close()
        run_log_closed["value"] = True

    # ------------------------------------------------------------------ fonts
    root = tk.Tk()
    root.title("Run All  /  VOD Highlight Pipeline")
    root.minsize(1000, 660)
    root.configure(bg=GUI_BG)
    try:
        root.state("zoomed")  # the concept is drawn for a maximized 1920x1080
    except Exception:
        root.geometry("1440x900")
    try:
        root.attributes("-alpha", 0.0)  # faded in by fade_window_in()
    except tk.TclError:
        pass

    display_family = _resolve_font_family(
        root,
        ["Helvetica Compressed", "Helvetica Compressed Bold", "Arial Narrow", "Haettenschweiler", "Impact"],
    )
    body_family = _resolve_font_family(root, ["Helvetica", "Arial", "Segoe UI"])

    FONT_TITLE = tkfont.Font(family=display_family, size=30, weight="bold")
    FONT_PANEL_TITLE = tkfont.Font(family=body_family, size=10, weight="bold")
    FONT_BODY = tkfont.Font(family=body_family, size=11)
    FONT_BODY_BOLD = tkfont.Font(family=body_family, size=11, weight="bold")
    FONT_BODY_SMALL = tkfont.Font(family=body_family, size=10)
    FONT_STATUS_CODE = tkfont.Font(family=display_family, size=24, weight="bold")
    FONT_BUTTON = tkfont.Font(family=display_family, size=13, weight="bold")
    FONT_MAP_ST = tkfont.Font(family=display_family, size=19, weight="bold")
    FONT_MAP_SYNC = tkfont.Font(family=display_family, size=14, weight="bold")
    FONT_MAP_FINISH = tkfont.Font(family=display_family, size=20, weight="bold")
    FONT_MAP_LABEL = tkfont.Font(family=body_family, size=12)
    FONT_CONSOLE = tkfont.Font(family=body_family, size=10)

    # ------------------------------------------------------------- run state
    # Terminal box modes ("finished" / "failed" / "stopped") freeze the status
    # panel so late mini events can't overwrite them.
    box_mode = {"value": "live"}  # live | finished | failed | stopped | sleep-cancelled
    current_box_step = {"index": None}

    step_state: list[str] = ["Waiting"] * len(steps)
    mini_state: dict[tuple[int, str], str] = {}
    mini_detail: dict[tuple[int, str], str] = {}
    sync_state: list[str] = [GUI_PROGRESS_RED] * (len(steps) - 1)
    finish_lit = {"value": False}
    step_runtime: list[dict[str, str]] = []

    is_single_track = _vod_is_single_track(target_folder) is True
    route_name = "TWITCH VOD" if is_single_track else "LOCAL VOD"

    for index, _step in enumerate(steps):
        process_detail, model_detail = _step_model_details(index, target_folder)
        step_runtime.append(
            {
                "task": "",
                "model": model_detail,
                "message": process_detail,
                "next": _initial_next_detail(index, target_folder),
                "stage": "",
            }
        )
        for position, (code, _name) in enumerate(_mini_stages_for_step(index, target_folder)):
            mini_state[(index, code)] = "Waiting"
            waiting_for = f"Waiting for Step {index + 1}" if position == 0 else f"Waiting for {_mini_stages_for_step(index, target_folder)[position - 1][0]}"
            mini_detail[(index, code)] = waiting_for

    # ---------------------------------------------------------------- layout
    main_frame = tk.Frame(root, bg=GUI_BG, padx=16, pady=10)
    main_frame.pack(fill="both", expand=True)
    main_frame.columnconfigure(0, weight=1, uniform="cols")
    main_frame.columnconfigure(1, weight=1, uniform="cols")
    main_frame.rowconfigure(1, weight=1)

    # -------------------------------------------------------------- header
    header_panel = tk.Frame(main_frame, bg=GUI_BG)
    header_panel.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
    header_panel.columnconfigure(0, weight=1)

    header_box = tk.Frame(header_panel, bg=GUI_BG)
    header_box.grid(row=0, column=0, sticky="ew")
    tk.Label(
        header_box,
        text=f"POG ENGINE: {RUN_ALL_VERSION} - {route_name}",
        anchor="w",
        bg=GUI_BG,
        fg=GUI_WHITE,
        font=FONT_TITLE,
    ).pack(anchor="w")
    tk.Label(
        header_box,
        text=f"{base_name}  *  {target_folder}",
        anchor="w",
        bg=GUI_BG,
        fg=GUI_ORANGE,
        font=FONT_BODY,
    ).pack(anchor="w", pady=(0, 2))

    stop_button = _GhostButton(header_panel, "STOP RUN", font=FONT_BUTTON, padx=18, pady=8, ring=2)
    stop_button.grid(row=0, column=1, sticky="ne")

    def _disable_stop(text: str, color: str = GUI_PROGRESS_RED) -> None:
        stop_button.configure(state="disabled", text=text, disabledforeground=color, ring_color=color)

    # ---------------------------------------------------------- left region
    left_region = tk.Frame(main_frame, bg=GUI_BG)
    left_region.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
    left_region.columnconfigure(0, weight=5, uniform="left")
    left_region.columnconfigure(1, weight=4, uniform="left")
    left_region.rowconfigure(1, weight=1)

    console_wrap, console_inner, console_title_label = _make_titled_panel(
        left_region, "LIVE CONSOLE: 0 LINES",
        bg=GUI_BG, border=GUI_WHITE, title_fg=GUI_ORANGE, title_font=FONT_PANEL_TITLE,
    )
    console_wrap.grid(row=0, column=0, sticky="nsew")

    log_box = tk.Text(
        console_inner,
        height=9,
        wrap="word",
        state="disabled",
        bg="#000000",
        fg=GUI_WHITE,
        insertbackground=GUI_WHITE,
        selectbackground=GUI_PROGRESS_RED,
        selectforeground=GUI_WHITE,
        relief="flat",
        padx=8,
        pady=4,
        bd=0,
        font=FONT_CONSOLE,
    )
    log_box.tag_configure("info", foreground=GUI_WHITE)
    log_box.tag_configure("success", foreground=GUI_NEON_GREEN)
    log_box.tag_configure("failure", foreground=GUI_PURE_RED)
    log_box.tag_configure("warning", foreground=GUI_ORANGE)
    log_box.pack(fill="both", expand=True)
    tk.Label(
        console_inner,
        text="FULL LOG SAVED TO step6_run_*.log",
        bg=GUI_BG,
        fg=GUI_ORANGE,
        font=FONT_BODY_SMALL,
    ).pack(fill="x", pady=(2, 2))

    status_wrap, status_inner, status_title_label = _make_titled_panel(
        left_region, "STANDBY",
        bg=GUI_BG, border=GUI_WHITE, title_fg=GUI_ORANGE, title_font=FONT_PANEL_TITLE,
    )
    status_wrap.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

    status_code_label = tk.Label(status_inner, text="", bg=GUI_BG, fg=GUI_WHITE, font=FONT_STATUS_CODE)
    status_code_label.pack(anchor="w")
    status_name_label = tk.Label(status_inner, text="", bg=GUI_BG, fg=GUI_WHITE, font=FONT_BODY_BOLD)
    status_name_label.pack(anchor="w")
    status_state_label = tk.Label(status_inner, text="", bg=GUI_BG, fg=GUI_WHITE, font=FONT_BODY_BOLD)
    status_state_label.pack(anchor="w")
    status_detail_label = tk.Label(
        status_inner, text="", bg=GUI_BG, fg=GUI_WHITE, font=FONT_BODY_SMALL,
        wraplength=380, justify="left", anchor="w",
    )
    status_detail_label.pack(anchor="w", fill="x", pady=(2, 0))
    status_github_label = tk.Label(
        status_inner, text="", bg=GUI_BG, fg=GUI_PURE_RED, font=FONT_BODY_SMALL,
        wraplength=380, justify="left", anchor="w",
    )
    status_github_label.pack(anchor="w", fill="x", pady=(0, 2))

    map_wrap, map_inner, _map_title = _make_titled_panel(
        left_region, None,
        bg=GUI_BG, border=GUI_WHITE, title_fg=GUI_ORANGE, title_font=FONT_PANEL_TITLE,
    )
    map_wrap.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(10, 0))

    # --------------------------------------------------------- right region
    right_area = tk.Frame(main_frame, bg=GUI_BG)
    right_area.grid(row=1, column=1, sticky="nsew")
    right_area.columnconfigure(0, weight=1)

    gallery_wrap, gallery_inner, _gallery_title = _make_titled_panel(
        right_area, "IMAGE GALLERY",
        bg=GUI_BG, border=GUI_WHITE, title_fg=GUI_ORANGE, title_font=FONT_PANEL_TITLE,
    )
    gallery_wrap.grid(row=0, column=0, sticky="nsew")
    saver_wrap, saver_inner, _saver_title = _make_titled_panel(
        right_area, "SCREEN SAVER",
        bg=GUI_BG, border=GUI_WHITE, title_fg=GUI_ORANGE, title_font=FONT_PANEL_TITLE,
    )
    # `saver_wrap` is mapped only while the gallery is hidden (restored window).
    right_area.rowconfigure(0, weight=1)  # the visible panel owns the full column

    # ----------------------------------------------------------------- map
    map_canvas = tk.Canvas(map_inner, bg=GUI_BG, highlightthickness=0, bd=0)
    map_canvas.pack(fill="both", expand=True, padx=6, pady=6)

    map_ids: dict[str, int] = {}
    cell_geometry: dict[str, dict[str, object]] = {}
    glow_geo: dict[str, list[tuple[float, float]]] = {}
    glow_keys: set[str] = set()
    glow_halo_ids: dict[str, list[int]] = {}
    # Glow = pre-rendered Gaussian-blur RGBA sprites, cross-faded through the
    # intensity tables below. Tk canvas has no per-item alpha, so blurred
    # PhotoImage sprites are the only clean way to do this; they build one per
    # event-loop tick so the UI never freezes. Intensity is per state: done
    # and lit cells breathe gently, the in-progress cell throbs hard and
    # shifts green-to-green, failed throbs red.
    GLOW_MARGIN = 44
    GLOW_PHASES_DEFAULT = (0.8, 0.95, 1.0, 0.95)
    GLOW_PHASES_RUNNING = (0.3, 0.7, 1.0, 0.7)
    GLOW_PHASES_FAILED = (0.35, 0.75, 1.0, 0.75)
    glow_sprite_cache: dict[tuple, tuple] = {}
    glow_build_queue: list[tuple] = []
    glow_builder_scheduled = {"value": False}
    glow_clock = {"phase": 0, "running": False}
    intro_done = {"value": False}

    def _short_label(index: int) -> str:
        raw = steps[index].label
        return raw.split(". ", 1)[1] if ". " in raw else raw

    def _text_color_for_fill(fill: str) -> str:
        # Concept close-ups: bar text stays black in EVERY state (neon, both
        # greens, both reds); mini code labels are orange and live outside.
        return GUI_TEXT_DARK

    def _mini_key(index: int, code: str) -> str:
        return f"mini:{index}:{code}"

    def _ribbon(x: float, y: float, w: float, h: float, cap: float, mirror: bool = False) -> list[tuple[float, float]]:
        """Mini cell from the concept close-ups: a diagonal ribbon with
        vertical end caps of height `cap` and ~45-degree top/bottom edges,
        inside bounding box (x, y, w, h) with h = w + cap. Columns 1/3/5 rise
        "/" (topmost at top-right); columns 2/4 fall "\\" mirrored."""
        if mirror:
            return [
                (x, y),                # top-left (topmost point)
                (x, y + cap),          # left cap bottom
                (x + w, y + h),        # bottom-right (lowest point)
                (x + w, y + h - cap),  # right cap top
            ]
        return [
            (x + w, y),            # top-right (topmost point)
            (x + w, y + cap),      # right cap bottom
            (x, y + h),            # bottom-left (lowest point)
            (x, y + h - cap),      # left cap top
        ]

    def _chip_ribbon(cell_x: float, cell_y: float, cell_w: float, cell_h: float, cell_cap: float, mirror: bool = False) -> list[tuple[float, float]]:
        """The orange accent, redesigned to sit INSIDE the cell (updated
        close-ups): a strip of the ribbon's own band hugging the upper
        diagonal from the left edge to a vertical cut at 0.27 * width.
        Rising cells hug the diagonal at the left cap's top corner; falling
        cells hug it at the topmost corner."""
        x_cut = cell_x + cell_w * 0.27
        t = cell_w * 0.14
        drop = x_cut - cell_x
        if mirror:
            # falling: the upper diagonal descends from the top-left corner
            y0 = cell_y
            return [(cell_x, y0), (x_cut, y0 + drop), (x_cut, y0 + drop + t), (cell_x, y0 + t)]
        # rising: the upper diagonal rises from the left cap's top corner
        lt_y = cell_y + cell_h - cell_cap
        return [(cell_x, lt_y), (x_cut, lt_y - drop), (x_cut, lt_y - drop + t), (cell_x, lt_y + t)]

    def _inflate(points: list[tuple[float, float]], d: float) -> list[tuple[float, float]]:
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        grown = []
        for x, y in points:
            dx, dy = x - cx, y - cy
            length = max(math.hypot(dx, dy), 0.001)
            scale = (length + d) / length
            grown.append((cx + dx * scale, cy + dy * scale))
        return grown

    def _flat(points: list[tuple[float, float]]) -> list[float]:
        return [coordinate for point in points for coordinate in point]

    def _map_layout(width: int, height: int) -> dict[str, float]:
        pad_x = 26.0
        cols = len(steps)
        col_width = (width - 2 * pad_x) / cols
        st_w = min(col_width * 0.62, 235.0)
        st_h = 52.0
        st_gap = 24.0
        cell_w = max(60.0, min(col_width * 0.50, 128.0))
        cap = cell_w * 0.36
        cell_h = cell_w + cap
        rows = max(len(_mini_stages_for_step(i, target_folder)) for i in range(cols))
        pitch = cell_h * 0.66  # ribbon cells tuck into each other (concept map)
        sync_h = 50.0
        sync_w = min(col_width * 0.85, 245.0)
        sync_gap_above = 34.0
        finish_gap = 30.0
        finish_h = 56.0
        bottom_pad = 24.0
        # Scale the whole vertical rhythm so a tall box (maximized 1080p) fills
        # with map instead of dead space, and a short one stays legible.
        base_total = (
            st_h + st_gap + (rows - 1) * pitch + cell_h
            + sync_gap_above + sync_h + finish_gap + finish_h + bottom_pad
        )
        scale = max(0.55, min((height - 20.0) / base_total, 1.5))
        st_w *= scale
        st_h *= scale
        st_gap *= scale
        cell_w *= scale
        cap *= scale
        cell_h *= scale
        pitch *= scale
        sync_h *= scale
        sync_w *= scale
        sync_gap_above *= scale
        finish_gap *= scale
        finish_h *= scale
        total = (
            st_h + st_gap + (rows - 1) * pitch + cell_h
            + sync_gap_above + sync_h + finish_gap + finish_h + bottom_pad
        )
        y0 = max(14.0, (height - total) / 2)
        return {
            "pad_x": pad_x, "col_width": col_width, "st_w": st_w, "st_h": st_h,
            "st_gap": st_gap, "cell_w": cell_w, "cap": cap, "cell_h": cell_h,
            "pitch": pitch, "rows": float(rows), "sync_h": sync_h, "sync_w": sync_w,
            "sync_gap_above": sync_gap_above, "finish_gap": finish_gap,
            "finish_h": finish_h, "y0": y0,
        }

    def fade_wires(step_index: int = 0) -> None:
        """Fade the orange routing wires in alongside the first cells."""
        steps_total = 6
        try:
            map_canvas.itemconfigure("wire", fill=_blend_hex(GUI_BG, GUI_ORANGE, step_index / steps_total))
        except tk.TclError:
            return
        if step_index < steps_total:
            root.after(40, lambda: fade_wires(step_index + 1))

    def draw_map(_event: object | None = None) -> None:
        width = map_canvas.winfo_width()
        height = map_canvas.winfo_height()
        if width < 80 or height < 80:
            return
        map_canvas.delete("all")
        map_ids.clear()
        cell_geometry.clear()
        glow_geo.clear()
        glow_halo_ids.clear()
        # While the opening fade-in is pending, cells draw dark and are faded
        # in cell by cell by run_intro(); a resize redraw mid-intro re-draws
        # dark and the pending chains re-fade on the fresh items.
        dimmed = not intro_done["value"]
        layout = _map_layout(width, height)
        pad_x = layout["pad_x"]
        col_width = layout["col_width"]
        st_w = layout["st_w"]
        st_h = layout["st_h"]
        cell_w = layout["cell_w"]
        cap = layout["cap"]
        cell_h = layout["cell_h"]
        pitch = layout["pitch"]
        rows = int(layout["rows"])
        sync_h = layout["sync_h"]
        sync_w = layout["sync_w"]
        finish_h = layout["finish_h"]
        y0 = layout["y0"]
        y_first = y0 + st_h + layout["st_gap"]
        y_sync_top = y_first + (rows - 1) * pitch + cell_h + layout["sync_gap_above"]
        y_sync_mid = y_sync_top + sync_h / 2
        y_sync_bottom = y_sync_top + sync_h
        y_finish_top = y_sync_bottom + layout["finish_gap"]
        center_x = width / 2
        finish_w = min(width - 2 * pad_x - 20.0, width * 0.93)
        finish_left = center_x - finish_w / 2
        n = len(steps)
        cham = min(12.0, col_width * 0.08)

        def col_cx(i: int) -> float:
            return pad_x + col_width * (i + 0.5)

        def trunk_x(i: int) -> float:
            # label-stub bracket, right of every cell in column i
            return col_cx(i) + cell_w * 1.35

        def spine_x(i: int) -> float:
            # drops from the ST bar down through its cells; the LAST column's
            # spine runs down the RIGHT side and stops at its last cell
            return col_cx(i) + (cell_w * 0.26 if i == n - 1 else -cell_w * 0.22)

        def label_y(i: int, row: int) -> float:
            y = y_first + row * pitch
            if i % 2 == 1:  # falling columns label at their lower-right cap
                return y + cell_h - cap * 0.5
            return y + cap * 0.5

        # ---- routing wires first (orange circuit lines under everything)
        wires: list[list[tuple[float, float]]] = []
        st_mid = y0 + st_h / 2
        # ST bars chained to each other along the top (concept top band)
        for i in range(n - 1):
            wires.append([
                (col_cx(i) + st_w / 2, st_mid),
                (col_cx(i + 1) - st_w / 2, st_mid),
            ])
        for i in range(n):
            codes = _mini_stages_for_step(i, target_folder)
            sx = spine_x(i)
            if i < n - 1:
                # spine: ST bar down through the cells into the sync row
                wires.append([(sx, y0 + st_h + 2), (sx, y_sync_mid - cham)])
                sync_left = pad_x + col_width * (i + 1) - sync_w / 2
                dd = max(0.0, min(cham, sync_left - sx))
                if dd > 0:
                    wires.append([(sx, y_sync_mid - dd), (sx + dd, y_sync_mid)])
                wires.append([(sx + dd, y_sync_mid), (sync_left, y_sync_mid)])
                # label bracket: taps the ST chain, collects every label stub,
                # ends at the last label (the spine carries the flow onward)
                tx = trunk_x(i)
                wires.append([(tx, st_mid), (tx, label_y(i, len(codes) - 1))])
                for r in range(len(codes)):
                    stub_y = label_y(i, r)
                    wires.append([(col_cx(i) + cell_w * 0.5 + 30, stub_y), (tx, stub_y)])
            else:
                # last column: the spine feeds the last sync from the right -
                # down through the cells, elbow left above the sync, then a
                # 45-degree drop into the sync's top edge
                y_elbow = y_sync_top - cham
                x_target = pad_x + col_width * (n - 1)  # the last sync's center
                dd1 = max(0.0, min(cham, sx - x_target))
                wires.append([(sx, y0 + st_h + 2), (sx, y_elbow - dd1)])
                if dd1 > 0:
                    wires.append([(sx, y_elbow - dd1), (sx - dd1, y_elbow)])
                wires.append([(sx - dd1, y_elbow), (x_target + cham, y_elbow)])
                wires.append([(x_target + cham, y_elbow), (x_target, y_sync_top)])
        for j in range(n - 2):
            # sync-to-sync handoff line (only BETWEEN existing syncs - one
            # iteration too many here was the stray line past SYNC:4-5)
            sr = pad_x + col_width * (j + 1) + sync_w / 2
            sl_next = pad_x + col_width * (j + 2) - sync_w / 2
            wires.append([(sr, y_sync_mid), (sl_next, y_sync_mid)])
        for j in range(n - 1):
            # each sync drops into the finish bar with a 45-degree chamfer
            scx = pad_x + col_width * (j + 1)
            landing = finish_left + finish_w * (j + 1) / n
            dy = max(0.0, min(abs(landing - scx), 90.0))
            y1 = y_finish_top - 14 - dy
            pts = [(scx, y_sync_bottom), (scx, y1)]
            if dy > 2:
                pts.append((landing, y1 + dy))
            pts.append((landing, y_finish_top))
            wires.append(pts)
        wire_color = GUI_BG if dimmed else GUI_ORANGE
        for pts in wires:
            flat = [coordinate for point in pts for coordinate in point]
            map_canvas.create_line(*flat, fill=wire_color, width=2, tags=("wire",))

        # ---- big-step bars (rectangles with black text, left to right)
        for index in range(n):
            cx = col_cx(index)
            x = cx - st_w / 2
            key = f"st:{index}"
            state = step_state[index]
            map_ids[key] = map_canvas.create_rectangle(
                x, y0, x + st_w, y0 + st_h,
                fill=GUI_BG if dimmed else GUI_STATE_COLORS[state], outline="", tags=(key, "hover"),
            )
            map_canvas.create_text(
                cx, y0 + st_h / 2,
                text=f"ST-{index + 1}", font=FONT_MAP_ST,
                fill=GUI_BG if dimmed else _text_color_for_fill(state),
                tags=(key, "hover", "celltext"),
            )
            cell_geometry[key] = {"kind": "st", "index": index}
            glow_geo[key] = [(x, y0), (x + st_w, y0), (x + st_w, y0 + st_h), (x, y0 + st_h)]

        # ---- mini ribbon cells, stacked top-down per column (future additions
        #      append underneath): state fill + orange chip + orange code label.
        #      Odd-numbered columns rise "/", ST-2 and ST-4 fall "\" (concept).
        for index in range(n):
            codes = _mini_stages_for_step(index, target_folder)
            cx = col_cx(index)
            x = cx - cell_w / 2
            mirror = index % 2 == 1
            for row, (code, name) in enumerate(codes):
                y = y_first + row * pitch
                key = _mini_key(index, code)
                state = mini_state.get((index, code), "Waiting")
                fill = GUI_STATE_COLORS.get(state, GUI_PROGRESS_RED)
                map_ids[key] = map_canvas.create_polygon(
                    *_flat(_ribbon(x, y, cell_w, cell_h, cap, mirror)),
                    fill=GUI_BG if dimmed else fill, outline="", tags=(key, "hover"),
                )
                map_canvas.create_polygon(
                    *_flat(_chip_ribbon(x, y, cell_w, cell_h, cap, mirror)),
                    fill=GUI_BG if dimmed else GUI_ORANGE, outline="", tags=(key, "hover", "chip"),
                )
                map_canvas.create_text(
                    cx + cell_w * 0.5 + 8, label_y(index, row),
                    text=code.upper(), font=FONT_MAP_LABEL, anchor="w",
                    fill=GUI_BG if dimmed else GUI_ORANGE, tags=(key, "hover", "celllabel"),
                )
                cell_geometry[key] = {"kind": "mini", "index": index, "code": code, "name": name}
                glow_geo[key] = _ribbon(x, y, cell_w, cell_h, cap, mirror)

        # ---- sync bars on the column boundaries
        for j in range(n - 1):
            scx = pad_x + col_width * (j + 1)
            key = f"sync:{j}"
            cell_geometry[key] = {"kind": "sync", "index": j}
            color = _cell_fill_color(key)
            map_ids[key] = map_canvas.create_rectangle(
                scx - sync_w / 2, y_sync_top, scx + sync_w / 2, y_sync_bottom,
                fill=GUI_BG if dimmed else color, outline="", tags=(key, "hover"),
            )
            map_canvas.create_text(
                scx, y_sync_mid,
                text=f"SYNC:{j + 1}-{j + 2}", font=FONT_MAP_SYNC,
                fill=GUI_BG if dimmed else _text_color_for_fill(color),
                tags=(key, "hover", "celltext"),
            )
            glow_geo[key] = [
                (scx - sync_w / 2, y_sync_top), (scx + sync_w / 2, y_sync_top),
                (scx + sync_w / 2, y_sync_bottom), (scx - sync_w / 2, y_sync_bottom),
            ]

        # ---- finish bar
        key = "finish"
        color = GUI_NEON_GREEN if finish_lit["value"] else GUI_PROGRESS_RED
        map_ids[key] = map_canvas.create_rectangle(
            finish_left, y_finish_top, finish_left + finish_w, y_finish_top + finish_h,
            fill=GUI_BG if dimmed else color, outline="", tags=(key, "hover"),
        )
        map_canvas.create_text(
            center_x, y_finish_top + finish_h / 2,
            text="POG ENGINE FINISH", font=FONT_MAP_FINISH,
            fill=GUI_BG if dimmed else _text_color_for_fill(color),
            tags=(key, "hover", "celltext"),
        )
        cell_geometry[key] = {"kind": "finish"}
        glow_geo[key] = [
            (finish_left, y_finish_top), (finish_left + finish_w, y_finish_top),
            (finish_left + finish_w, y_finish_top + finish_h), (finish_left, y_finish_top + finish_h),
        ]

        # Recreate glow halos for every glowing cell at the current phase.
        glow_clock["phase"] = 0
        for key in list(glow_keys):
            _apply_glow(key)

        if dimmed:
            root.after(700, fade_wires)

        _bind_map_hover()

    def _cell_fill_color(key: str) -> str:
        geometry = cell_geometry.get(key)
        if geometry is None:
            return GUI_PROGRESS_RED
        kind = geometry["kind"]
        if kind == "st":
            if step_state[geometry["index"]] == "Running":
                return _running_fill()
            return GUI_STATE_COLORS[step_state[geometry["index"]]]
        if kind == "mini":
            state = mini_state.get((geometry["index"], geometry["code"]), "Waiting")
            if state == "Running":
                return _running_fill()
            return GUI_STATE_COLORS.get(state, GUI_PROGRESS_RED)
        if kind == "sync":
            state = sync_state[geometry["index"]]
            if state == "Running":
                return _running_fill()
            return state
        return GUI_NEON_GREEN if finish_lit["value"] else GUI_PROGRESS_RED

    def _glow_sprite(kind: str, w: float, h: float, color: str, phase: int, mirror: bool, state_tag: str = "done"):
        """Blurred RGBA glow halo in the cell's own silhouette, cached per
        (shape, size, color, phase). Building is queued one sprite per
        event-loop tick (a synchronous burst would freeze the UI); until a
        phase is built the closest cached phase is returned instead."""
        key = (kind, round(w), round(h), color, phase, mirror, state_tag)
        cached = glow_sprite_cache.get(key)
        if cached is not None:
            return cached
        if key not in glow_build_queue:
            glow_build_queue.append(key)
            _schedule_glow_builder()
        return _nearest_glow_sprite(kind, w, h, color, phase, mirror)

    def _nearest_glow_sprite(kind: str, w: float, h: float, color: str, phase: int, mirror: bool, state_tag: str = "done"):
        """Closest already-built phase for this cell shape, if any."""
        for offset in range(len(GLOW_PHASES_DEFAULT)):
            probe = (
                kind, round(w), round(h), color,
                (phase + offset) % len(GLOW_PHASES_DEFAULT), mirror, state_tag,
            )
            cached = glow_sprite_cache.get(probe)
            if cached is not None:
                return cached
        return None

    def _schedule_glow_builder() -> None:
        if glow_builder_scheduled["value"] or Image is None:
            return
        glow_builder_scheduled["value"] = True
        root.after(10, _glow_builder_tick)

    def _glow_builder_tick() -> None:
        glow_builder_scheduled["value"] = False
        if not glow_build_queue:
            return
        key = glow_build_queue.pop(0)
        if key not in glow_sprite_cache:
            _build_glow_sprite(key)
        if glow_build_queue:
            _schedule_glow_builder()

    def _build_glow_sprite(key: tuple) -> None:
        kind, wr, hr, color, phase, mirror, state_tag = key
        m = GLOW_MARGIN
        fw, fh = wr + 2 * m, hr + 2 * m
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        if kind == "mini":
            cap = max(4.0, wr * 0.36)
            if mirror:
                local = [(0, 0), (0, cap), (wr, hr), (wr, hr - cap)]
            else:
                local = [(wr, 0), (wr, cap), (0, hr), (0, hr - cap)]
        else:
            local = [(0, 0), (wr, 0), (wr, hr), (0, hr)]
        factor = {"run": GLOW_PHASES_RUNNING, "fail": GLOW_PHASES_FAILED}.get(
            state_tag, GLOW_PHASES_DEFAULT
        )[phase % 4]
        photos = []
        # three stacked layers: wide bloom + dense mid + hot white-tinted core
        # rim; the low gammas pack the falloff dense so the ring reads bright
        for blur, gamma, white in ((18, 0.5, 0.0), (8, 0.38, 0.15), (3, 0.55, 0.45)):
            layer_rgb = tuple(int(c + (255 - c) * white) for c in rgb)
            img = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.polygon([(m + px, m + py) for px, py in local], fill=layer_rgb + (255,))
            img = img.filter(ImageFilter.GaussianBlur(blur))
            alpha = img.getchannel("A").point(
                lambda a: int(255 * ((a / 255) ** gamma) * factor)
            )
            img.putalpha(alpha)
            photos.append(ImageTk.PhotoImage(img))
        glow_sprite_cache[key] = tuple(photos)
        # a freshly built phase can complete halos that were waiting on it
        for glow_key in list(glow_keys):
            if glow_key not in glow_halo_ids:
                _apply_glow(glow_key)

    def _running_fill() -> str:
        """The in-progress cell's fill: cycling from Progress Green to Neon
        Green and back on the pulse clock (settles on Neon when it finishes)."""
        factor = GLOW_PHASES_RUNNING[glow_clock["phase"] % 4]
        t = max(0.0, min(1.0, (factor - 0.3) / 0.7))
        return _blend_hex(GUI_PROGRESS_GREEN, GUI_NEON_GREEN, t)

    def _glow_state_tag(key: str) -> str:
        geometry = cell_geometry.get(key)
        if geometry is None:
            return "done"
        kind = geometry["kind"]
        if kind == "mini":
            state = mini_state.get((geometry["index"], geometry["code"]), "Waiting")
        elif kind == "st":
            state = step_state[geometry["index"]]
        elif kind == "sync":
            state = (
                "Running" if sync_state[geometry["index"]] == "Running"
                else "Done" if sync_state[geometry["index"]] == GUI_NEON_GREEN
                else "Waiting"
            )
        else:
            state = "Done" if finish_lit["value"] else "Waiting"
        return {"Running": "run", "Failed": "fail", "Stopped": "stop"}.get(state, "done")

    def _glow_color_for(key: str) -> str:
        # the running cell's glow follows its animated fill (quantized so the
        # sprite cache stays small)
        geometry = cell_geometry.get(key)
        if geometry is not None:
            is_running = (
                mini_state.get((geometry["index"], geometry["code"]), "Waiting") == "Running"
                if geometry["kind"] == "mini"
                else (
                    step_state[geometry["index"]] == "Running"
                    if geometry["kind"] == "st"
                    else (
                        sync_state[geometry["index"]] == "Running"
                        if geometry["kind"] == "sync"
                        else False
                    )
                )
            )
            if is_running:
                factor = GLOW_PHASES_RUNNING[glow_clock["phase"] % 4]
                t = max(0.0, min(1.0, (factor - 0.3) / 0.7))
                return _blend_hex(GUI_PROGRESS_GREEN, GUI_NEON_GREEN, round(t * 4) / 4)
        return _cell_fill_color(key)

    def _apply_glow(key: str) -> None:
        old = glow_halo_ids.pop(key, [])
        for item_id in old:
            try:
                map_canvas.delete(item_id)
            except tk.TclError:
                pass
        color = _glow_color_for(key)
        if color == GUI_PROGRESS_RED or key not in map_ids:
            glow_keys.discard(key)
            return
        glow_keys.add(key)
        if not intro_done["value"]:
            return
        points = glow_geo.get(key)
        geometry = cell_geometry.get(key)
        if not points or geometry is None or Image is None or ImageTk is None:
            return
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        minx, miny = min(xs), min(ys)
        w = max(xs) - minx
        h = max(ys) - miny
        mirror = geometry["index"] % 2 == 1 if geometry["kind"] == "mini" else False
        state_tag = _glow_state_tag(key)
        photos = _glow_sprite(geometry["kind"], w, h, color, glow_clock["phase"], mirror, state_tag)
        if photos is None:
            return  # still building; the builder re-applies glow once ready
        halo_ids = []
        for photo in photos:
            halo_id = map_canvas.create_image(
                minx - GLOW_MARGIN, miny - GLOW_MARGIN, image=photo, anchor="nw",
            )
            map_canvas.tag_lower(halo_id)
            halo_ids.append(halo_id)
        glow_halo_ids[key] = halo_ids

    def _apply_glow_phase(key: str, phase: int) -> None:
        halo_ids = glow_halo_ids.get(key, [])
        if not halo_ids:
            return
        color = _glow_color_for(key)
        geometry = cell_geometry.get(key)
        points = glow_geo.get(key)
        if color == GUI_PROGRESS_RED or geometry is None or not points or Image is None:
            return
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        mirror = geometry["index"] % 2 == 1 if geometry["kind"] == "mini" else False
        state_tag = _glow_state_tag(key)
        photos = _glow_sprite(
            geometry["kind"], max(xs) - min(xs), max(ys) - min(ys), color, phase, mirror, state_tag,
        )
        if photos is None:
            return  # phase still building; keep showing the last phase
        try:
            for halo_id, photo in zip(halo_ids, photos):
                map_canvas.itemconfigure(halo_id, image=photo)
        except tk.TclError:
            pass

    def glow_tick() -> None:
        if not glow_clock["running"]:
            return
        glow_clock["phase"] = (glow_clock["phase"] + 1) % len(GLOW_PHASES_DEFAULT)
        for key in list(glow_halo_ids):
            _apply_glow_phase(key, glow_clock["phase"])
        # the in-progress mini cell shifts its own fill green-to-green on the
        # same clock (its glow follows via _glow_color_for)
        for (mini_index, code), mini_st in list(mini_state.items()):
            if mini_st != "Running":
                continue
            fill_key = _mini_key(mini_index, code)
            item_id = map_ids.get(fill_key)
            if item_id is not None:
                try:
                    map_canvas.itemconfigure(item_id, fill=_cell_fill_color(fill_key))
                except tk.TclError:
                    pass
        for step_index, step_st in list(enumerate(step_state)):
            if step_st != "Running":
                continue
            fill_key = f"st:{step_index}"
            item_id = map_ids.get(fill_key)
            if item_id is not None:
                try:
                    map_canvas.itemconfigure(item_id, fill=_cell_fill_color(fill_key))
                except tk.TclError:
                    pass
        for sync_index, sync_st in list(enumerate(sync_state)):
            if sync_st != "Running":
                continue
            fill_key = f"sync:{sync_index}"
            item_id = map_ids.get(fill_key)
            if item_id is not None:
                try:
                    map_canvas.itemconfigure(item_id, fill=_cell_fill_color(fill_key))
                except tk.TclError:
                    pass
        root.after(480, glow_tick)

    def start_glow() -> None:
        if glow_clock["running"]:
            return
        glow_clock["running"] = True
        glow_tick()

    def _paint_cell(key: str) -> None:
        item_id = map_ids.get(key)
        if item_id is None:
            return
        fill = _cell_fill_color(key)
        try:
            map_canvas.itemconfigure(item_id, fill=fill)
            text_color = _text_color_for_fill(fill)
            for text_id in map_canvas.find_withtag(key):
                if "celltext" in map_canvas.gettags(text_id):
                    map_canvas.itemconfigure(text_id, fill=text_color)
        except tk.TclError:
            return
        _apply_glow(key)

    map_resize_after = {"id": None}

    def on_map_resize(event: object) -> None:
        if getattr(event, "widget", None) is not map_canvas:
            return
        if map_resize_after["id"] is not None:
            root.after_cancel(map_resize_after["id"])
        map_resize_after["id"] = root.after(140, draw_map)

    map_canvas.bind("<Configure>", on_map_resize)

    # ---------------------------------------------------------- the tooltip
    tooltip: dict[str, object] = {"window": None, "after": None, "key": None}
    hover_outline = {"key": None}

    def _set_hover_outline(key: str | None) -> None:
        """Outline the hovered cell in orange (cleared on leave)."""
        previous = hover_outline["key"]
        if previous is not None and previous in map_ids:
            try:
                map_canvas.itemconfigure(map_ids[previous], outline="", width=1)
            except tk.TclError:
                pass
        hover_outline["key"] = key
        if key is not None and key in map_ids:
            try:
                map_canvas.itemconfigure(map_ids[key], outline=GUI_ORANGE, width=2)
            except tk.TclError:
                pass

    def hide_tooltip(_event: object | None = None) -> None:
        _set_hover_outline(None)
        if tooltip["after"] is not None:
            root.after_cancel(tooltip["after"])
            tooltip["after"] = None
        tooltip["key"] = None
        if tooltip["window"] is not None:
            try:
                tooltip["window"].destroy()
            except tk.TclError:
                pass
            tooltip["window"] = None

    def _tooltip_state_color(state: str) -> str:
        color = GUI_STATE_COLORS.get(state, GUI_WHITE)
        # progress red on the black tooltip is unreadable; waiting states read white
        return GUI_WHITE if color == GUI_PROGRESS_RED else color

    def tooltip_body(key: str) -> tuple[str, list[tuple[str, str]]]:
        geometry = cell_geometry[key]
        kind = geometry["kind"]
        if kind == "st":
            index = geometry["index"]
            state = step_state[index]
            process_detail, model_detail = _step_model_details(index, target_folder)
            lines = [
                (f"ST-{index + 1}  -  {_short_label(index).upper()}", GUI_ORANGE),
                (state.upper(), _tooltip_state_color(state)),
                (process_detail, GUI_WHITE),
            ]
            if model_detail:
                lines.append((model_detail, GUI_WHITE))
            lines.append((step_runtime[index]["next"] or step_runtime[index]["message"], GUI_ORANGE))
            return "POG ENGINE STEP", lines
        if kind == "mini":
            index = geometry["index"]
            code = geometry["code"]
            state = mini_state.get((index, code), "Waiting")
            detail = mini_detail.get((index, code), "")
            lines = [
                (f"{code.upper()}  -  {geometry['name'].upper()}", GUI_ORANGE),
                (state.upper(), _tooltip_state_color(state)),
                (mini_description(index, code, target_folder), GUI_WHITE),
            ]
            stage_label = step_runtime[index]["stage"]
            if index == 4 and stage_label and stage_label.casefold() in ANALYSIS_STAGE_BY_LABEL:
                info = ANALYSIS_STAGE_DETAILS[stage_label]
                if info["model"]:
                    lines.append((info["model"], GUI_ORANGE))
            if detail:
                lines.append((detail, GUI_ORANGE))
            return "MINI-PROCESS", lines
        if kind == "sync":
            j = geometry["index"]
            lit = sync_state[j] in {GUI_NEON_GREEN, "Running"}
            cycling = sync_state[j] == "Running"
            state_word = "CYCLING" if cycling else ("LIT" if lit else "WAITING")
            lines = [
                (f"SYNC:{j + 1}-{j + 2}", GUI_ORANGE),
                (state_word, GUI_NEON_GREEN if lit else GUI_WHITE),
                (
                    f"Lights when Step {j + 1} finishes and Step {j + 2} starts - "
                    "proof the handoff between the two stages happened.",
                    GUI_WHITE,
                ),
            ]
            return "HANDOFF", lines
        lit = finish_lit["value"]
        lines = [
            ("POG ENGINE FINISH", GUI_ORANGE),
            ("LIT" if lit else "WAITING", GUI_NEON_GREEN if lit else GUI_WHITE),
            ("Turns green only when every step and every sync has finished.", GUI_WHITE),
        ]
        return "THE FINISH LINE", lines

    def show_tooltip(key: str) -> None:
        if tooltip["key"] != key or tooltip["window"] is not None:
            return
        title_text, lines = tooltip_body(key)
        window = tk.Toplevel(root)
        window.overrideredirect(True)
        window.attributes("-topmost", True)
        window.configure(bg=GUI_BG, highlightthickness=2, highlightbackground=GUI_WHITE)
        # Withdraw until it has its final geometry: mapping at the default
        # position briefly puts it under the pointer, which fires a canvas
        # <Leave> and kills the tooltip before it ever shows.
        window.withdraw()
        inner = tk.Frame(window, bg=GUI_BG)
        inner.pack(fill="both", expand=True, padx=2, pady=2)
        tk.Label(inner, text=title_text, bg=GUI_BG, fg=GUI_ORANGE, font=FONT_PANEL_TITLE).pack(
            anchor="w", padx=10, pady=(8, 2)
        )
        for text, color in lines:
            if not text:
                continue
            tk.Label(
                inner, text=text, bg=GUI_BG, fg=color, font=FONT_BODY_SMALL,
                wraplength=360, justify="left", anchor="w",
            ).pack(anchor="w", padx=10, pady=(0, 2))
        tk.Label(inner, text=" ", bg=GUI_BG, fg=GUI_BG, font=FONT_BODY_SMALL).pack()
        window.update_idletasks()
        tip_w = max(window.winfo_reqwidth(), 240)
        tip_h = window.winfo_reqheight()
        x = min(root.winfo_pointerx() + 18, root.winfo_screenwidth() - tip_w - 8)
        y = min(root.winfo_pointery() + 18, root.winfo_screenheight() - tip_h - 8)
        window.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        window.deiconify()
        window.lift()
        tooltip["window"] = window

    def on_cell_enter(key: str, _event: object = None) -> None:
        hide_tooltip()
        _set_hover_outline(key)
        tooltip["key"] = key
        tooltip["after"] = root.after(260, lambda: show_tooltip(key))

    def on_cell_move(_event: object) -> None:
        window = tooltip["window"]
        if window is None:
            return
        tip_w = max(window.winfo_reqwidth(), 240)
        tip_h = window.winfo_reqheight()
        x = min(root.winfo_pointerx() + 18, root.winfo_screenwidth() - tip_w - 8)
        y = min(root.winfo_pointery() + 18, root.winfo_screenheight() - tip_h - 8)
        window.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    def _bind_map_hover() -> None:
        for key in cell_geometry:
            def enter(_event: object, k: str = key) -> None:
                on_cell_enter(k)

            map_canvas.tag_bind(key, "<Enter>", enter)
            map_canvas.tag_bind(key, "<Leave>", hide_tooltip)
            map_canvas.tag_bind(key, "<Motion>", on_cell_move)

    # ------------------------------------------------------- the status box
    def set_status_box(
        title: str,
        code_text: str,
        name_text: str,
        state_word: str,
        state_color: str,
        detail_text: str,
        *,
        show_github: bool = False,
    ) -> None:
        status_title_label.configure(text=f" {title} ")
        status_code_label.configure(text=code_text)
        status_name_label.configure(text=name_text)
        status_state_label.configure(text=state_word, fg=state_color)
        status_detail_label.configure(text=detail_text)
        status_github_label.configure(
            text="Please check the log and raise an issue on GitHub" if show_github else ""
        )

    def _active_mini(index: int) -> tuple[str, str]:
        """The mini cell that represents this step right now (running first,
        then the most significant non-waiting state, then simply the first)."""
        codes = _mini_stages_for_step(index, target_folder)
        for phase in ("Running", "Failed", "Stopped", "Done", "Skipped"):
            for code, _name in codes:
                if mini_state.get((index, code)) == phase:
                    return code, phase
        return codes[0][0], "Waiting"

    def refresh_status_box() -> None:
        if box_mode["value"] != "live":
            return
        index = current_box_step["index"]
        if index is None:
            set_status_box("STANDBY", "", "", "", GUI_WHITE, "Warming up the pipeline...")
            return
        code, state = _active_mini(index)
        name = dict(_mini_stages_for_step(index, target_folder)).get(code, "")
        state_words = {
            "Waiting": "WAITING", "Running": "WORKING", "Done": "DONE",
            "Skipped": "DONE", "Failed": "FAILED", "Stopped": "STOPPED",
        }
        runtime = step_runtime[index]
        detail = runtime["message"] or runtime["task"] or ""
        if len(detail) > 220:
            detail = detail[:217] + "..."
        set_status_box(
            f"STEP {index + 1}: {_short_label(index).upper()}",
            code.upper(), name.upper(),
            state_words.get(state, state.upper()),
            GUI_STATE_COLORS.get(state, GUI_WHITE),
            detail,
            show_github=(state == "Failed"),
        )

    # ------------------------------------------------------------ the console
    console_line_count = {"value": 0}

    def append_log(text: str, tag: str | None = None) -> None:
        console_line_count["value"] += text.count("\n") or 1
        if console_line_count["value"] == 1 or console_line_count["value"] % 25 == 0:
            console_title_label.configure(text=f" LIVE CONSOLE: {console_line_count['value']:,} LINES ")
        log_box.configure(state="normal")
        log_box.insert("end", text, tag or _console_log_tag(text))
        log_box.see("end")
        log_box.configure(state="disabled")
        write_run_log(text)

    # ------------------------------------------------------------ the gallery
    gallery_paths = gallery_image_paths()
    gallery_index = {"value": 0}
    gallery_photo = {"value": None}
    current_gallery_path = {"value": None}
    gallery_render_after = {"id": None}
    gallery_rotation_after = {"id": None}
    gallery_meta_var = tk.StringVar(value="")
    gallery_visible = {"value": True}

    image_label = tk.Label(
        gallery_inner,
        text=(
            f"IMAGE GALLERY\n\nNo images found in:\n{GALLERY_DIR}"
            if not gallery_paths
            else "Loading gallery images..."
        ),
        anchor="center",
        justify="center",
        bg=GUI_BG,
        fg=GUI_ORANGE if not gallery_paths else GUI_WHITE,
        font=FONT_TITLE if not gallery_paths else FONT_BODY,
    )
    image_label.pack(fill="both", expand=True, pady=(4, 2))

    gallery_footer = tk.Frame(gallery_inner, bg=GUI_BG)
    gallery_footer.pack(fill="x", side="bottom")

    def _ghost_button(parent: object, text: str, command, *, font, padx=14, pady=5) -> _GhostButton:
        return _GhostButton(parent, text, command, font=font, padx=padx, pady=pady, ring=1)

    tk.Label(
        gallery_footer,
        textvariable=gallery_meta_var,
        anchor="w",
        bg=GUI_BG,
        fg=GUI_ORANGE,
        font=FONT_BODY_SMALL,
    ).pack(side="left", fill="x", expand=True)

    def render_gallery_image(image_path: Path) -> None:
        if not gallery_visible["value"]:
            return
        try:
            available_width = max(image_label.winfo_width() - 16, 240)
            available_height = max(image_label.winfo_height() - 16, 240)
            if Image is not None and ImageTk is not None:
                with Image.open(image_path) as source_image:
                    scale = min(
                        available_width / max(source_image.width, 1),
                        available_height / max(source_image.height, 1),
                        1.0,
                    )
                    width = max(round(source_image.width * scale), 1)
                    height = max(round(source_image.height * scale), 1)
                    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                    image = source_image.resize((width, height), resample)
                gallery_photo["value"] = ImageTk.PhotoImage(image)
                image_label.configure(image=gallery_photo["value"], text="")
            else:
                gallery_photo["value"] = tk.PhotoImage(file=str(image_path))
                image_label.configure(image=gallery_photo["value"], text="")
            gallery_meta_var.set(
                f"{gallery_index['value'] + 1} / {len(gallery_paths)}  *  {image_path.name}"
            )
        except Exception as exc:
            image_label.configure(text=f"Could not load gallery image:\n{exc}", image="")
            gallery_photo["value"] = None

    def show_gallery_image(offset: int = 0) -> None:
        if not gallery_paths:
            gallery_meta_var.set("Gallery folder is empty or unavailable")
            return
        gallery_index["value"] = (gallery_index["value"] + offset) % len(gallery_paths)
        current_gallery_path["value"] = gallery_paths[gallery_index["value"]]
        render_gallery_image(current_gallery_path["value"])

    def rotate_gallery() -> None:
        if gallery_visible["value"] and len(gallery_paths) > 1:
            show_gallery_image(1)
        gallery_rotation_after["id"] = root.after(8000, rotate_gallery)

    def rerender_gallery_image(_event: object | None = None) -> None:
        if not gallery_visible["value"] or current_gallery_path["value"] is None:
            return
        if gallery_render_after["id"] is not None:
            root.after_cancel(gallery_render_after["id"])
        gallery_render_after["id"] = root.after(
            150,
            lambda: render_gallery_image(current_gallery_path["value"]),
        )

    image_label.bind("<Configure>", rerender_gallery_image)

    def open_screensaver() -> None:
        if not gallery_paths or Image is None or ImageTk is None:
            return
        saver = tk.Toplevel(root)
        saver.geometry(f"{saver.winfo_screenwidth()}x{saver.winfo_screenheight()}+0+0")
        saver.configure(bg="black")
        saver.attributes("-topmost", True)
        saver_label = tk.Label(saver, bg="black", cursor="none")
        saver_label.place(relx=0.5, rely=0.5, anchor="center")
        saver_state = {"index": gallery_index["value"], "after": None, "photo": None}

        def render_saver_image() -> None:
            path = gallery_paths[saver_state["index"] % len(gallery_paths)]
            try:
                with Image.open(path) as source_image:
                    screen_w = max(saver.winfo_width(), saver.winfo_screenwidth())
                    screen_h = max(saver.winfo_height(), saver.winfo_screenheight())
                    scale = min(
                        screen_w / max(source_image.width, 1),
                        screen_h / max(source_image.height, 1),
                    )
                    width = max(round(source_image.width * scale), 1)
                    height = max(round(source_image.height * scale), 1)
                    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                    saver_state["photo"] = ImageTk.PhotoImage(
                        source_image.resize((width, height), resample)
                    )
                saver_label.configure(image=saver_state["photo"])
            except Exception:
                pass

        def advance_saver() -> None:
            saver_state["index"] += 1
            render_saver_image()
            saver_state["after"] = saver.after(8000, advance_saver)

        def close_saver(_event: object | None = None) -> None:
            if saver_state["after"] is not None:
                saver.after_cancel(saver_state["after"])
            saver.destroy()

        saver.bind("<Escape>", close_saver)
        saver.bind("<Button-1>", close_saver)
        saver.bind("<Button-3>", close_saver)
        saver.focus_force()
        render_saver_image()
        saver_state["after"] = saver.after(8000, advance_saver)

    screensaver_disabled = not gallery_paths or Image is None or ImageTk is None

    saver_button = _ghost_button(
        gallery_footer, "VIEW FULLSCREEN", open_screensaver, font=FONT_PANEL_TITLE, padx=12, pady=3,
    )
    saver_button.pack(side="right", padx=(4, 6))
    if screensaver_disabled:
        saver_button.configure(state="disabled")

    # The fallback button lives in its own right-column panel while the
    # gallery is hidden (restored window).
    saver_only_button = _ghost_button(
        saver_inner, "SCREEN SAVER MODE", open_screensaver, font=FONT_BUTTON, padx=22, pady=10,
    )
    saver_only_button.pack(pady=(14, 10), padx=20, fill="x")
    tk.Label(
        saver_inner,
        text=(
            "THE BEST-OF GALLERY RUNS FULLSCREEN UNTIL POG ENGINE FINISHES"
            if gallery_paths
            else f"NO IMAGES FOUND IN:\n{GALLERY_DIR}"
        ),
        bg=GUI_BG,
        fg=GUI_ORANGE,
        font=FONT_BODY_SMALL,
        wraplength=380,
        justify="center",
    ).pack(pady=(0, 18), padx=16)
    if screensaver_disabled:
        saver_only_button.configure(state="disabled")

    def apply_gallery_visibility() -> None:
        try:
            maximized = root.state() == "zoomed"
        except Exception:
            maximized = False
        should_show = maximized and root.winfo_width() >= 1500
        if should_show == gallery_visible["value"]:
            return
        gallery_visible["value"] = should_show
        if should_show:
            saver_wrap.grid_remove()
            gallery_wrap.grid(row=0, column=0, sticky="nsew")
            right_area.rowconfigure(0, weight=1)
            right_area.rowconfigure(1, weight=0)
            if current_gallery_path["value"] is not None:
                render_gallery_image(current_gallery_path["value"])
        else:
            gallery_wrap.grid_remove()
            saver_wrap.grid(row=0, column=0, sticky="nsew")
            right_area.rowconfigure(0, weight=0)
            right_area.rowconfigure(1, weight=1)

    def on_root_resize(event: object) -> None:
        # The toplevel bindtag fires this for every child Configure too; only
        # the root window's own resize can change the gallery layout.
        if getattr(event, "widget", None) is not root:
            return
        root.after_idle(apply_gallery_visibility)

    root.bind("<Configure>", on_root_resize)

    # ---------------------------------------------------- step / mini plumbing
    def set_mini_stage(index: int, code: str, status: str, detail: str) -> None:
        key = (index, code)
        if key not in mini_state:
            return
        mini_state[key] = status
        if detail:
            mini_detail[key] = detail
        _paint_cell(_mini_key(index, code))

    def set_step_mini(index: int, code: str, status: str, detail: str) -> None:
        stage_names = dict(_mini_stages_for_step(index, target_folder))
        if code not in stage_names:
            return
        step_runtime[index]["stage"] = f"{code}. {stage_names[code]}"
        set_mini_stage(index, code, status, detail)
        if step_state[index] == "Running":
            current_box_step["index"] = index
            refresh_status_box()

    def set_all_step_minis(index: int, status: str, detail: str) -> None:
        for code, _name in _mini_stages_for_step(index, target_folder):
            set_step_mini(index, code, status, detail)

    def begin_step_minis(index: int) -> None:
        first_code, _name = _mini_stages_for_step(index, target_folder)[0]
        set_step_mini(index, first_code, "Running", "Starting mini-process")

    def _mark_active_mini(index: int, status: str) -> None:
        """On a big-step failure/stop, push that state onto the live mini cell."""
        codes = _mini_stages_for_step(index, target_folder)
        target = None
        for code, _name in codes:
            if mini_state.get((index, code)) == "Running":
                target = code
                break
        if target is None:
            for code, _name in codes:
                if mini_state.get((index, code)) not in {"Waiting", "Done", "Skipped"}:
                    target = code
                    break
        if target is not None:
            set_mini_stage(index, target, status, status.upper())

    def set_step_status(index: int, value: str) -> None:
        step_state[index] = value
        if value == "Running":
            begin_step_minis(index)
        elif value in {"Done", "Skipped"}:
            set_all_step_minis(index, value, "Big-step output is ready")
            step_runtime[index]["next"] = ""
        elif value == "Failed":
            _mark_active_mini(index, "Failed")
        elif value == "Stopped":
            _mark_active_mini(index, "Stopped")
        update_syncs()
        update_finish()
        current_box_step["index"] = index
        _paint_cell(f"st:{index}")
        refresh_status_box()

    def update_syncs() -> None:
        """SYNC:j lights when step j finished and step j+1 started (concept rule).
        While step j+1 is still running, the lit sync cycles green-to-green
        like the in-progress mini cells; it settles solid neon when that step
        finishes."""
        for j in range(len(steps) - 1):
            prev_done = step_state[j] in {"Done", "Skipped"}
            next_started = step_state[j + 1] in {"Running", "Done", "Skipped", "Failed", "Stopped"}
            if not (prev_done and next_started):
                new_state = GUI_PROGRESS_RED
            elif step_state[j + 1] == "Running":
                new_state = "Running"
            else:
                new_state = GUI_NEON_GREEN
            if new_state != sync_state[j]:
                sync_state[j] = new_state
                _paint_cell(f"sync:{j}")

    def update_finish() -> None:
        lit = all(state in {"Done", "Skipped"} for state in step_state)
        if lit != finish_lit["value"]:
            finish_lit["value"] = lit
            _paint_cell("finish")

    def update_step_detail(index: int, message: str) -> None:
        step_runtime[index]["message"] = message
        if current_box_step["index"] == index:
            refresh_status_box()

    def render_step_detail(index: int) -> None:
        if current_box_step["index"] == index:
            refresh_status_box()

    def update_step_from_output(index: int, raw_line: str) -> None:
        line = _strip_console_ansi(raw_line)
        if not line:
            return
        lowered = line.casefold()
        state = step_runtime[index]

        if index == 0:
            if _vod_is_single_track(target_folder) is True:
                if "step 1a: extracting full mix" in lowered:
                    state["next"] = "Next: Split the full mix into Demucs input chunks"
                    set_step_mini(index, "1a", "Running", line[:120])
                elif "step 1b/1c:" in lowered:
                    state["next"] = "Next: Run Demucs on every saved input chunk"
                    set_step_mini(index, "1a", "Done", "Full mixed audio extracted")
                    set_step_mini(index, "1b", "Running", line[:120])
                elif "extracting demucs chunk" in lowered:
                    state["next"] = "Next: Finish preparing Demucs input chunks"
                    set_step_mini(index, "1b", "Running", line[:120])
                elif "demucs input chunks ready" in lowered:
                    state["next"] = "Next: Separate each Demucs input chunk"
                    set_step_mini(index, "1b", "Done", "Demucs input chunks and manifest saved")
                    set_step_mini(index, "1c", "Running", "Starting per-chunk Demucs separation")
                elif "running:" in lowered and "demucs" in lowered:
                    set_step_mini(index, "1c", "Running", line[:120])
                elif "mic chunks ready" in lowered:
                    state["next"] = "Next: Combine separated mic chunks"
                    set_step_mini(index, "1c", "Done", "Every input chunk separated")
                    set_step_mini(index, "1d", "Done", "Separated mic chunks and manifest saved")
                elif "combined separated mic audio" in lowered:
                    state["next"] = "Next: Render the final 16 kHz mono mic track"
                    set_step_mini(index, "1e", "Done", "Separated mic chunks combined")
                elif "saved rendered voice track" in lowered:
                    state["next"] = "Next: Prepare the Whisper chunk set"
                    set_step_mini(index, "1f", "Done", "Rendered 16 kHz mono mic WAV saved")
                elif "preparing " in lowered:
                    state["next"] = "Next: Finish saving the Whisper chunk set"
                    set_step_mini(index, "1g", "Running", line[:120])
                elif "finished preparing " in lowered:
                    state["next"] = ""
                    set_step_mini(index, "1g", "Done", "Whisper chunks and timing manifest saved")
                elif "done! rendered mic audio and saved whisper chunks" in lowered:
                    state["next"] = ""
                    set_all_step_minis(index, "Done", "Rendered mic WAV and Whisper chunks ready")
            else:
                if "separate game/mic audio tracks" in lowered:
                    state["next"] = "Next: Extract, then split and save all mic chunks"
                    set_step_mini(index, "1a", "Running", "ffmpeg is extracting track 1")
                elif "step 1a:" in lowered or "step 1a complete" in lowered:
                    state["next"] = "Next: Split and save all mic chunks"
                    set_step_mini(index, "1a", "Running", line[:120])
                elif "step 1b/1c:" in lowered or "preparing " in lowered:
                    state["next"] = "Next: Finish saving the chunk set"
                    set_step_mini(index, "1a", "Done", "Mic audio extraction complete")
                    set_step_mini(index, "1b", "Running", line[:120])
                elif "extracting chunk" in lowered or "using preserved audio chunk:" in lowered:
                    set_step_mini(index, "1b", "Running", line[:120])
                elif "finished preparing " in lowered:
                    state["next"] = ""
                    set_step_mini(index, "1a", "Done", "Mic audio extracted")
                    set_step_mini(index, "1b", "Done", "All audio chunks split")
                    set_step_mini(index, "1c", "Done", "Chunk files and timing manifest saved")
                elif "done! mic audio and saved chunks" in lowered:
                    state["next"] = ""
                    set_all_step_minis(index, "Done", "Mic WAV and saved chunks ready")
        elif index == 1:
            if "transcribing " in lowered and "saved chunk(s)" in lowered:
                state["next"] = "Next: Finish Whisper for every saved chunk"
                set_step_mini(index, "2a", "Running", line[:120])
            elif "running whisper for chunk" in lowered:
                state["next"] = "Next: Process the next saved chunk"
                set_step_mini(index, "2a", "Running", line[:120])
            elif "all chunk srts finished" in lowered:
                state["next"] = "Next: Stitch all chunk SRTs"
                set_step_mini(index, "2a", "Done", "Every chunk SRT completed")
                set_step_mini(index, "2b", "Running", "Combining shifted timestamps")
            elif "stitched " in lowered and "subtitle block" in lowered:
                state["next"] = ""
                set_step_mini(index, "2a", "Done", "Every chunk SRT completed")
                set_step_mini(index, "2b", "Done", line[:120])
        elif index == 2:
            if "fixing srt" in lowered:
                state["next"] = "Next: Repair timestamps and remove adjacent repeats"
                set_step_mini(index, "3a", "Done", "SRT input loaded")
                set_step_mini(index, "3b", "Running", "Checking timestamp order")
            elif "removed " in lowered and "repeated" in lowered:
                set_step_mini(index, "3b", "Done", "Timestamp pass complete")
                set_step_mini(index, "3c", "Done", line[:120])
            elif "fixed file saved" in lowered:
                state["next"] = ""
                set_all_step_minis(index, "Done", "Fixed SRT ready")
        elif index == 3:
            if "splitting srt" in lowered:
                state["next"] = "Next: Regroup captions into transcript parts"
                set_step_mini(index, "4a", "Done", "Fixed SRT input loaded")
                set_step_mini(index, "4b", "Running", "Merging captions into thought blocks")
            elif "created:" in lowered:
                state["next"] = "Next: Write transcript_part files"
                set_step_mini(index, "4b", "Done", "Thought blocks merged")
                set_step_mini(index, "4c", "Done", "Time ranges assigned")
                set_step_mini(index, "4d", "Running", "Writing transcript_part files")
            elif "chunk length:" in lowered:
                state["next"] = ""
                set_all_step_minis(index, "Done", "Transcript parts ready")

        if index == 1:
            if "transcribing " in lowered and "saved chunk(s)" in lowered:
                state["task"] = "Running Whisper once per saved chunk"
                state["message"] = line[:220]
            elif "running whisper for chunk" in lowered:
                state["task"] = "Running fresh Whisper process for chunk"
                state["message"] = line[:220]
            elif "all chunk srts finished" in lowered:
                state["task"] = "Stitching chunk transcripts"
                state["message"] = line[:220]
            elif "stitched " in lowered and "subtitle block" in lowered:
                state["task"] = "Connected SRT timestamps written"
                state["message"] = line[:220]

        if index == 4:
            for stage_label, stage_info in ANALYSIS_STAGE_DETAILS.items():
                if stage_label.casefold() not in lowered:
                    continue
                state["stage"] = stage_label
                code, _name = ANALYSIS_STAGE_BY_LABEL[stage_label.casefold()]
                if "checkpoint already exists, skipping" in lowered:
                    state["task"] = f"Finished earlier: {stage_info['task']}"
                    state["message"] = "Checkpoint found; this mini-process resumed without rerunning."
                    set_mini_stage(index, code, "Skipped", "Checkpoint reused")
                else:
                    state["task"] = f"Running mini-process: {stage_info['task']}"
                    state["message"] = "Reading live analyzer output..."
                    set_mini_stage(index, code, "Running", "Live output received")
                state["model"] = stage_info["model"]
                state["next"] = stage_info["next"]
                break

            if "stage finished in " in lowered and state["stage"]:
                code, _name = ANALYSIS_STAGE_BY_LABEL[state["stage"].casefold()]
                set_mini_stage(index, code, "Done", line[:74])
            elif "speech emotion scoring" in lowered:
                state["task"] = "Running mini-process: Emotion Scoring"
                state["model"] = f"Model: {EMOTION_MODEL_ID}" if EMOTION_ENABLED else "Model: disabled by configuration"
            elif "full-file audio scan" in lowered:
                state["task"] = "Running mini-process: Audio Scan"
                state["model"] = f"Title model when needed: {JUDGE_MODEL}"
            elif "verifying " in lowered:
                state["task"] = "Running mini-process: Verification"
                state["model"] = f"Model: {JUDGE_MODEL}"
            elif "running judge stage" in lowered or "judging in " in lowered:
                state["task"] = "Running mini-process: Judging"
                state["model"] = f"Model: {JUDGE_MODEL}"
            elif "loading ollama model" in lowered or (
                "ollama model" in lowered and " ready in " in lowered
            ) or "llama-server model ready" in lowered:
                # Cold model load; surface it on the active mini-process cell
                # instead of looking like a silent hang.
                if state["stage"]:
                    mini_code, _mini_name = ANALYSIS_STAGE_BY_LABEL[state["stage"].casefold()]
                    set_mini_stage(index, mini_code, "Running", line[:120])
            elif "top " in lowered and " highlights saved" in lowered:
                state["task"] = "Running mini-process: Export"
                state["model"] = "Model: none"

        model_match = re.search(
            r"(Model(?: metadata source)?|VAD model):\s*(.+)",
            line,
            flags=re.IGNORECASE,
        )
        if model_match:
            model_label, model_value = model_match.groups()
            state["model"] = f"Loaded {model_label}: {model_value.strip()}"

        action_patterns = (
            r"^Step \d+/\d+:",
            r"^Transcribing ",
            r"^Fixing ",
            r"^Splitting ",
            r"Analyzing transcript_part",
            r"Generating titles for ",
            r"Saved \d+ emotion score",
            r"Response received in ",
            r"^Found \d+ transcript part",
            r"^Running judge stage",
        )
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in action_patterns):
            state["task"] = line[:180]
        elif "found output:" in lowered:
            state["message"] = line[:220]
        elif "stage finished in " in lowered:
            state["message"] = line[:180]
        elif line.startswith("[") and ("[" in line[1:] or "]" in line):
            state["message"] = f"Latest: {line[:180]}"
        render_step_detail(index)

    # --------------------------------------------------------------- worker
    def run_step_process(index: int, step: RunAllStep) -> bool:
        step_started_perf = time.perf_counter()
        step_started_at = datetime.now()
        duration_recorded = False

        def record_duration(
            status: str,
            *,
            elapsed_override: float | None = None,
            note: str = "",
        ) -> None:
            nonlocal duration_recorded
            if duration_recorded:
                return
            finished_at = datetime.now()
            elapsed = (
                elapsed_override
                if elapsed_override is not None
                else time.perf_counter() - step_started_perf
            )
            record_big_step_duration(
                run_id=run_id,
                target_folder=target_folder,
                step_number=index + 1,
                step_label=step.label,
                status=status,
                duration_seconds=elapsed,
                started_at=step_started_at,
                finished_at=finished_at,
            )
            suffix = f" {note}" if note else ""
            events.put(("log", f"[DURATION] {step.label}: {elapsed:.1f}s ({status}){suffix}\n"))
            duration_recorded = True

        step_path = target_folder / step.bat_name
        if not step_path.exists():
            record_duration("Failed")
            events.put(("status", (index, "Failed")))
            events.put(("failed", f"Missing step script: {step_path}"))
            return False

        if step.expected_kind is not None:
            existing_output, _ = run_all_file_info(target_folder, base_name, step.expected_kind)
            if existing_output is not None:
                previous_duration = latest_completed_step_duration(target_folder, index + 1)
                if previous_duration is None:
                    record_duration("Skipped", note="no previous completed duration")
                    previous_detail = "No previous completed duration was found."
                else:
                    record_duration(
                        "Skipped",
                        elapsed_override=previous_duration,
                        note="reused previous completed duration",
                    )
                    previous_detail = f"Previous completed duration: {previous_duration:.1f}s."
                events.put(("status", (index, "Skipped")))
                events.put((
                    "detail",
                    (index, f"Output already exists: {existing_output}. {previous_detail}"),
                ))
                events.put(("log", f"\n--- {step.label} ---\n"))
                events.put(("log", f"Skipping because output already exists: {existing_output}\n"))
                events.put(("log", f"{previous_detail}\n"))
                events.put(("progress", index + 1))
                return True

        if step.bat_name == "5_AnalyzeHighlights.bat":
            if LLM_BACKEND == "llamacpp":
                # Managed server: pipeline will start/stop llama-server itself and hot-swap
                # GGUFs between discovery and judge. Just verify the GGUF files exist;
                # don't require the server to already be running.
                def _resolve_llama(role: str) -> str:
                    if role == "discovery":
                        p = (LLAMA_DISCOVERY_MODEL_PATH or "").strip()
                        if p:
                            return p
                        return (LLAMA_MODEL_PATH or "").strip()
                    p = (LLAMA_JUDGE_MODEL_PATH or "").strip()
                    if p:
                        return p
                    p2 = (LLAMA_DISCOVERY_MODEL_PATH or "").strip()
                    if p2:
                        return p2
                    return (LLAMA_MODEL_PATH or "").strip()
                missing = []
                seen_paths: set[str] = set()
                for role in ("discovery", "judge"):
                    pp = _resolve_llama(role)
                    if pp in seen_paths:
                        continue
                    seen_paths.add(pp)
                    if not pp or not Path(pp).exists():
                        missing.append((role, pp or "(empty)"))
                if missing:
                    msg_lines = "\n".join(f"  - {role}: {pp}" for role, pp in missing)
                    message = f"llama.cpp GGUF not found for managed server:\n{msg_lines}\nSet LLAMA_DISCOVERY_MODEL_PATH / LLAMA_JUDGE_MODEL_PATH (or LLAMA_MODEL_PATH) in pipeline_config.py or the configurator."
                    record_duration("Failed")
                    events.put(("status", (index, "Failed")))
                    events.put(("detail", (index, "GGUF missing - cannot start managed llama-server.")))
                    events.put(("log", f"\n--- {step.label} ---\n{message}\n"))
                    events.put(("ollama_unavailable", message))
                    events.put(("failed", message))
                    return False
            else:
                base_url = ollama_base_url(OLLAMA_URL)
                if not ollama_is_reachable(base_url):
                    message = ollama_not_ready_message(base_url)
                    record_duration("Failed")
                    events.put(("status", (index, "Failed")))
                    events.put(("detail", (index, "Ollama must be running before Step 5 can start.")))
                    events.put(("log", f"\n--- {step.label} ---\n{message}\n"))
                    events.put(("ollama_unavailable", message))
                    events.put(("failed", message))
                    return False

        events.put(("status", (index, "Running")))
        events.put(("detail", (index, f"Started {step.bat_name}; waiting for its live output.")))
        events.put(("log", f"\n--- {step.label} ---\n"))

        args: list[str] = []
        if step.input_kind is not None:
            input_path, input_description = run_all_file_info(target_folder, base_name, step.input_kind)
            if input_path is None:
                record_duration("Failed")
                events.put(("status", (index, "Failed")))
                events.put(("failed", f"{step.label} could not find input file:\n{input_description}"))
                return False
            events.put(("detail", (index, f"Input: {input_path}\nRunning: {step.bat_name}")))
            events.put(("log", f"Using input: {input_path}\n"))
            if step.pass_input:
                args.append(str(input_path))

        command = ["cmd.exe", "/d", "/c", "call", str(step_path), *args]
        env = os.environ.copy()
        env["RUN_ALL"] = "1"
        # Keep every child Python process line-buffered even when the GUI is
        # launched with pythonw and the batch process has no console window.
        env["PYTHONUNBUFFERED"] = "1"
        events.put(("detail", (index, f"Subprocess: {step.bat_name}\nCommand started; live output is being forwarded below.")))
        events.put(("log", f"[LIVE] Starting subprocess: {' '.join(command)}\n"))

        try:
            process = subprocess.Popen(
                command,
                cwd=target_folder,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:
            record_duration("Failed")
            events.put(("status", (index, "Failed")))
            events.put(("failed", f"{step.label} could not start: {exc}"))
            return False
        current_process["popen"] = process
        assert process.stdout is not None
        try:
            while True:
                line = process.stdout.readline()
                if line:
                    events.put(("output", (index, line)))
                    continue
                if process.poll() is not None:
                    break
                time.sleep(0.01)
        finally:
            process.stdout.close()

        return_code = process.wait()
        current_process["popen"] = None

        if stop_requested["value"]:
            record_duration("Stopped")
            events.put(("status", (index, "Stopped")))
            return False

        if return_code:
            record_duration("Failed")
            events.put(("status", (index, "Failed")))
            events.put(("failed", f"{step.label} failed with exit code {return_code}."))
            return False

        if step.expected_kind is not None:
            expected_output, expected_description = run_all_file_info(
                target_folder, base_name, step.expected_kind
            )
            if expected_output is None:
                record_duration("Failed")
                events.put(("status", (index, "Failed")))
                events.put(("failed", f"{step.label} did not create expected file:\n{expected_description}"))
                return False
            events.put(("detail", (index, f"Output verified: {expected_output}")))
            events.put(("log", f"Found output: {expected_output}\n"))

        record_duration("Done")
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

    def request_system_sleep() -> None:
        """Put Windows to sleep after a successful, opted-in pipeline run.

        Waits AUTO_SLEEP_DELAY_SECONDS first. Any mouse movement or
        keystroke during that countdown cancels the sleep entirely, so an
        actively used PC stays awake and the GUI returns to its Done state.
        """
        delay_seconds = max(0, int(AUTO_SLEEP_DELAY_SECONDS))
        if delay_seconds > 0:
            events.put((
                "log",
                f"[sleep] PC will sleep in {delay_seconds}s. "
                "Move the mouse or press any key to cancel.\n",
            ))
            start_tick = _last_input_tick()
            deadline = time.monotonic() + delay_seconds
            while time.monotonic() < deadline:
                tick = _last_input_tick()
                if (
                    start_tick is not None
                    and tick is not None
                    and ((tick - start_tick) & 0xFFFFFFFF) != 0
                ):
                    events.put(("sleep_cancelled", None))
                    return
                time.sleep(0.5)
        try:
            import ctypes

            set_suspend_state = ctypes.windll.powrprof.SetSuspendState
            set_suspend_state.argtypes = [ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
            set_suspend_state.restype = ctypes.c_bool
            if not set_suspend_state(False, False, False):
                raise OSError("SetSuspendState returned FALSE")
        except Exception as exc:
            events.put(("log", f"[sleep] Could not put the PC to sleep: {exc}\n"))

    def unload_ollama_models_now() -> None:
        if LLM_BACKEND == "llamacpp":
            events.put(("log", "[stop] llama-server keeps model resident (llamacpp backend - no unload).\n"))
            return
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
            if run_all_file_info(target_folder, base_name, step.expected_kind)[0] is not None:
                last_done_label = step.label
        return last_done_label

    def on_stop_clicked() -> None:
        if stop_requested["value"]:
            return  # already stopping, ignore extra clicks
        stop_requested["value"] = True
        _disable_stop("STOPPING...")
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

    # --------------------------------------------------------- event draining
    def drain_events() -> None:
        processed = 0
        while processed < 300:
            try:
                kind, payload = events.get_nowait()
            except queue.Empty:
                break
            processed += 1
            if kind == "status":
                index, value = payload
                set_step_status(index, value)
                if value == "Running":
                    open_step_log(index)
                elif value in {"Done", "Failed", "Stopped", "Skipped"}:
                    close_step_log()
                append_log(
                    f"[STATUS] {steps[index].label}: {value}\n",
                    "success" if value in {"Done", "Skipped"} else _console_log_tag(value),
                )
            elif kind == "detail":
                index, message = payload
                update_step_detail(index, str(message))
            elif kind == "ollama_unavailable":
                messagebox.showwarning("Ollama is not running", str(payload), parent=root)
            elif kind == "output":
                index, line = payload
                append_log(line)
                write_step_log(index, line)
                update_step_from_output(index, line)
            elif kind == "log":
                append_log(payload)
            elif kind == "progress":
                pass  # the pipeline map itself is the progress display
            elif kind == "failed":
                exit_code["value"] = 1
                box_mode["value"] = "failed"
                index = current_box_step["index"] if current_box_step["index"] is not None else 0
                code, _state = _active_mini(index)
                name = dict(_mini_stages_for_step(index, target_folder)).get(code, "")
                set_status_box(
                    f"STEP {index + 1}: {_short_label(index).upper()}",
                    code.upper(), name.upper(), "FAILED", GUI_PURE_RED, str(payload),
                    show_github=True,
                )
                append_log(f"\nERROR: {payload}\n", "failure")
                _disable_stop("STOPPED", GUI_PURE_RED)
            elif kind == "stopped":
                box_mode["value"] = "stopped"
                last_done = payload
                set_status_box(
                    "STOPPED BY USER", "", "", "STOPPED", GUI_ORANGE,
                    f"Last fully completed step: {last_done}. "
                    "Run Run_Pog_Engine.bat again to continue from there.",
                )
                append_log(
                    f"\nStopped by user. Last fully completed step: {last_done}.\n"
                    "Ollama models unloaded. Run Run_Pog_Engine.bat again to continue.\n",
                    "warning",
                )
                _disable_stop("STOPPED", GUI_ORANGE)
            elif kind == "sleep_cancelled":
                box_mode["value"] = "sleep-cancelled"
                set_status_box(
                    "ALL STEP FINISHED", "", "", "DONE", GUI_NEON_GREEN,
                    "Auto-sleep cancelled by mouse/keyboard activity.",
                )
                append_log(
                    "[sleep] Input detected during the countdown - auto-sleep cancelled.\n",
                    "warning",
                )
            elif kind == "complete":
                box_mode["value"] = "finished"
                status_message = str(payload)
                if AUTO_SLEEP_AFTER_PIPELINE:
                    status_message += f" PC will go to sleep in {AUTO_SLEEP_DELAY_SECONDS}s (input cancels it)."
                set_status_box(
                    "ALL STEP FINISHED", "", "", "DONE", GUI_NEON_GREEN,
                    (
                        "POG ENGINE HAS FINISHED ALL STEPS"
                        if not AUTO_SLEEP_AFTER_PIPELINE
                        else status_message
                    ),
                )
                append_log(f"\n{status_message}\n", "success")
                write_run_log(f"[PROGRESS] {len(steps)}/{len(steps)}\n")
                _disable_stop("FINISHED", GUI_NEON_GREEN)
                if AUTO_SLEEP_AFTER_PIPELINE:
                    append_log(
                        f"[sleep] Auto-sleep is enabled; suspending Windows in "
                        f"{AUTO_SLEEP_DELAY_SECONDS}s unless you move the mouse or type.\n",
                        "success",
                    )
                    threading.Thread(target=request_system_sleep, daemon=True).start()

        root.after(80, drain_events)

    # ------------------------------------------------------- opening fade-in
    intro_cells: list[str] = [f"st:{index}" for index in range(len(steps))]
    for index in range(len(steps)):
        for code, _name in _mini_stages_for_step(index, target_folder):
            intro_cells.append(_mini_key(index, code))
    intro_cells.extend(f"sync:{j}" for j in range(len(steps) - 1))
    intro_cells.append("finish")

    def fade_cell(key: str, step_index: int = 0) -> None:
        steps_total = 6
        fill = _cell_fill_color(key)
        t = step_index / steps_total
        try:
            for item_id in map_canvas.find_withtag(key):
                tags = map_canvas.gettags(item_id)
                if item_id == map_ids.get(key):
                    target = fill
                elif "chip" in tags or "celllabel" in tags:
                    target = GUI_ORANGE
                else:
                    target = GUI_TEXT_DARK  # bar text: black in every state
                map_canvas.itemconfigure(item_id, fill=_blend_hex(GUI_BG, target, t))
        except tk.TclError:
            return
        if step_index < steps_total:
            root.after(36, lambda: fade_cell(key, step_index + 1))
        else:
            _paint_cell(key)

    def run_intro() -> None:
        # windows -> pieces -> cells: panels reveal in sequence, then every
        # map cell fades in one by one, then the glow starts breathing.
        panels = [header_panel, console_wrap, status_wrap, map_wrap, right_area]
        for position, panel in enumerate(panels):
            root.after(120 + position * 110, lambda p=panel: p.grid())
        base_delay = 120 + len(panels) * 110 + 60
        for cell_position, key in enumerate(intro_cells):
            root.after(base_delay + cell_position * 45, lambda k=key: fade_cell(k))
        root.after(base_delay + len(intro_cells) * 45 + 260, _intro_complete)

    def _intro_complete() -> None:
        intro_done["value"] = True
        # A mid-intro redraw paints dark; make sure every cell ends at its
        # final state color, then start the glow breathing.
        for key in list(cell_geometry):
            _paint_cell(key)
        start_glow()

    def fade_window_in(sequence: int = 0) -> None:
        alpha_values = (0.25, 0.5, 0.72, 0.88, 1.0)
        try:
            if sequence >= len(alpha_values):
                root.attributes("-alpha", 1.0)
                return
            root.attributes("-alpha", alpha_values[sequence])
        except tk.TclError:
            return
        root.after(45, lambda: fade_window_in(sequence + 1))

    def on_window_close() -> None:
        glow_clock["running"] = False
        close_step_log()
        if gallery_render_after["id"] is not None:
            root.after_cancel(gallery_render_after["id"])
        if gallery_rotation_after["id"] is not None:
            root.after_cancel(gallery_rotation_after["id"])
        if map_resize_after["id"] is not None:
            root.after_cancel(map_resize_after["id"])
        hide_tooltip()
        close_run_log()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_window_close)

    # Hide the panels immediately; run_intro() reveals them one by one while
    # the window itself fades up from alpha 0.
    for _panel in (header_panel, console_wrap, status_wrap, map_wrap, right_area):
        _panel.grid_remove()

    show_gallery_image()
    if len(gallery_paths) > 1:
        gallery_rotation_after["id"] = root.after(8000, rotate_gallery)

    # One-time layout migration for pre-redesign VOD folders.
    migrate_legacy_vod_folder(target_folder, events)

    threading.Thread(target=worker, daemon=True).start()
    root.after(80, drain_events)
    root.after(120, fade_window_in)
    root.after(220, run_intro)
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
    step5_dir = step_subdir(target_folder, 5)
    step5_dir.mkdir(parents=True, exist_ok=True)
    bat_files = {
        "1_ExtractMicAudio.bat": make_extract_mic_bat(target_folder, base_name, video_suffix, is_single_track),
        "2_TranscribeAudio.bat": make_transcribe_bat(script_path),
        "3_FixSRT.bat": make_fix_srt_bat(script_path, target_folder),
        "4_SplitSRT.bat": make_split_srt_bat(script_path, target_folder),
        "5_AnalyzeHighlights.bat": make_analyze_bat(target_folder),
        "Run_Pog_Engine.bat": make_run_all_bat(target_folder, base_name, script_path),
        "Start_LlamaServer.bat": make_llama_server_bat(),
        # Debug-only sub-steps live inside the step 5 folder, next to the
        # checkpoints and logs they operate on. Not in the main numbered
        # sequence or tracked by the RunAll GUI.
        "5a_Discovery.bat": make_debug_stage_bat(target_folder, "discovery", "Discovery"),
        "5b_AudioScan.bat": make_debug_stage_bat(target_folder, "audioscan", "Audio Scan"),
        "5c_EmotionScoring.bat": make_debug_stage_bat(target_folder, "emotion", "Emotion Scoring"),
        "5d_Verify.bat": make_debug_stage_bat(target_folder, "verify", "Verification"),
        "5e_Judge.bat": make_debug_stage_bat(target_folder, "judge", "Judging"),
        "5f_Export.bat": make_debug_stage_bat(target_folder, "export", "Export"),
    }

    for name, content in bat_files.items():
        out_path = step5_dir / name if name.startswith("5") and name[1].isalpha() else target_folder / name
        write_text_crlf(out_path, content, encoding="ascii")

    print(f"Created helper scripts in '{target_folder}':")
    if is_single_track:
        print("   1_ExtractMicAudio.bat   <- isolate the voice, then split/save all chunks")
        print("                              (single-track Twitch-style VOD)")
    else:
        print("   1_ExtractMicAudio.bat   <- extract mic track, then split/save all chunks")
        print("                              (separate-track local recording)")
    print("   2_TranscribeAudio.bat   <- run Whisper on each saved chunk, then stitch the SRT")
    print("   3_FixSRT.bat            <- drag stitched .srt onto this to fix repeats/timestamps")
    print("   4_SplitSRT.bat          <- drag *_fixed.srt onto this to create transcript_part files")
    print("   5_AnalyzeHighlights.bat <- double-click for emotion-enhanced highlights")
    print("   Run_Pog_Engine.bat      <- double-click to run steps 1 through 5 in order")
    print("   Start_LlamaServer.bat   <- manual llama-server launcher (optional, pipeline manages hot-swap in llamacpp mode)")
    print("\n   Debug only (not part of the main sequence, not tracked by RunAll):")
    print("   5a_Discovery.bat        <- force-rerun just the LLM discovery passes")
    print("   5b_AudioScan.bat        <- force-rerun just the full-file audio scan")
    print("   5c_EmotionScoring.bat   <- force-rerun just the speech-emotion model")
    print("   5d_Verify.bat           <- force-rerun just the content verification pass")
    print("   5e_Judge.bat            <- force-rerun just the final ranking")
    print("   5f_Export.bat           <- rewrite the CSV + Resolve EDL from the last judged result")
    print("   (5a-5f debug bats are inside step5_analyze_highlights/)")
    print("\nStep 5 checkpoints internally after each of its sub-stages, so if it dies or")
    print("you hit Stop in the RunAll GUI partway through, running it again (or")
    print("Run_Pog_Engine.bat) picks up exactly where it stopped instead of starting over.")
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
    action_group.add_argument("--config", action="store_true", help="Open the model-configuration GUI (configure_models.py) to pick AI models / presets and write them to pipeline_config.py. No video file needed.")
    action_group.add_argument(
        "--prepare-audio-chunks",
        metavar="AUDIO",
        help="Step 1: split mic audio into overlapping chunks and save them.",
    )
    action_group.add_argument(
        "--transcribe-audio",
        metavar="AUDIO",
        help="Step 2: run Whisper on saved chunks and stitch their SRTs.",
    )
    parser.add_argument("--base-name", help="Video base name for --run-all-gui. Defaults to the folder name.")
    parser.add_argument("--chunk-minutes", type=int, default=DEFAULT_CHUNK_MINUTES, help="Transcript chunk length for --split-srt.")
    parser.add_argument("--transcription-chunk-minutes", type=int, default=TRANSCRIPTION_CHUNK_MINUTES, help="Whisper audio chunk length for --prepare-audio-chunks.")
    parser.add_argument("--transcription-overlap-seconds", type=int, default=TRANSCRIPTION_CHUNK_OVERLAP_SECONDS, help="Whisper audio overlap for --prepare-audio-chunks.")
    parser.add_argument("--out-dir", metavar="DIR", help="Output folder for --fix-srt / --split-srt results; the generated step bats bake in the VOD folder or step subfolder.")

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
        if args.prepare_audio_chunks:
            prepare_audio_chunks(
                Path(args.prepare_audio_chunks),
                args.transcription_chunk_minutes,
                args.transcription_overlap_seconds,
            )
            return 0

        if args.transcribe_audio:
            transcribe_audio_in_chunks(Path(args.transcribe_audio))
            return 0

        if args.fix_srt:
            out_dir = Path(args.out_dir) if args.out_dir else None
            fix_srt(Path(args.fix_srt), out_dir)
            return 0

        if args.split_srt:
            out_dir = Path(args.out_dir) if args.out_dir else None
            split_srt_into_chunks(Path(args.split_srt), args.chunk_minutes, out_dir)
            return 0

        if args.run_all_gui:
            target_folder = Path(args.run_all_gui)
            base_name = args.base_name or target_folder.name
            return run_all_gui(target_folder, base_name)

        if args.config:
            # Import lazily so a config edit never loads the heavyweight
            # pipeline imports (requests, the analyze_highlights module,
            # etc.) - configure_models.py only needs pipeline_config + tkinter.
            import configure_models
            return configure_models.run_gui()
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