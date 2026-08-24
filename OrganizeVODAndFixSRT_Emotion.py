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
    BIG_STEP_LABELS,
    EMOTION_ENABLED,
    EMOTION_MODEL_ID,
    JUDGE_MODEL,
    MODEL,
    NOISE_GATE_ATTACK_MS,
    NOISE_GATE_RATIO,
    NOISE_GATE_RELEASE_MS,
    NOISE_GATE_THRESHOLD_DB,
    OLLAMA_URL,
    STEP_HISTORY_FILENAME,
    TRANSCRIPTION_CHUNK_MINUTES,
    TRANSCRIPTION_CHUNK_OVERLAP_SECONDS,
    VOCAL_ISOLATION_MODEL,
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


def _ollama_base_url(url: str) -> str:
    return url.split("/api/", 1)[0]


def ollama_is_reachable(base_url: str, timeout: float = 3) -> bool:
    """Return true only when the Ollama health endpoint responds successfully."""
    try:
        response = requests.get(base_url.rstrip("/") + "/api/version", timeout=timeout)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        return False


def ollama_not_ready_message(base_url: str) -> str:
    return (
        f"Ollama is not active at {base_url}.\n"
        "Close the RunAll GUI, launch Ollama manually, wait for it to finish "
        "starting, then run 6_RunAllSteps.bat again."
    )

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
    for chunk_number, chunk_audio_path in enumerate(chunk_audio_paths, start=1):
        chunk_srt_path = chunk_audio_path.with_suffix(".srt")
        print(
            f"Running Whisper for chunk "
            f"{chunk_number:02d}/{len(chunk_audio_paths):02d}: {chunk_audio_path.name}",
            flush=True,
        )
        _transcribe_audio_chunk(chunk_audio_path, chunk_srt_path)
        chunk_srt_paths.append(chunk_srt_path)

    print("All chunk SRTs finished; stitching connected timestamps.", flush=True)
    output_path = audio_path.with_suffix(".srt")
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
    """Generate Step 1 for a local VOD with a separate mic track.

    Extraction, chunk splitting, and chunk persistence all finish here. Step 2
    only launches Whisper against the saved chunk files.
    """
    mic_wav_name = f"{base_name}_mic.wav"
    video_path = target_folder / f"{base_name}{video_suffix}"
    wav_path = target_folder / mic_wav_name
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
    combined_wav_name = f"{base_name}_mic_combined.w64"
    demucs_chunk_dir_name = f"{base_name}_demucs_chunks"
    mic_chunk_dir_name = f"{base_name}_mic_demucs_chunks"
    video_path = target_folder / f"{base_name}{video_suffix}"
    mixed_wav_path = target_folder / mixed_wav_name
    mic_wav_path = target_folder / mic_wav_name
    combined_wav_path = target_folder / combined_wav_name
    demucs_chunk_dir = target_folder / demucs_chunk_dir_name
    mic_chunk_dir = target_folder / mic_chunk_dir_name
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
echo Step 1e: combined separated audio saved as {combined_wav_name}
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

def _make_srt_step_bat(
    script_path: Path,
    *,
    missing_message: str,
    progress_message: str,
    command: str,
    error_label: str,
    done_message: str,
    next_message: str,
) -> str:
    return f'''@echo off
if "%~1"=="" (
    echo {missing_message}
    if not "%RUN_ALL%"=="1" pause
    exit /b 1
)

set SRT=%~1

echo {progress_message}: %~nx1
echo.
python "{batch_quote(script_path)}" {command} "%SRT%" --no-pause

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


def make_fix_srt_bat(script_path: Path) -> str:
    return _make_srt_step_bat(
        script_path,
        missing_message="Drag your .srt file onto this script.",
        progress_message="Fixing SRT timestamps and adjacent repeats",
        command="--fix-srt",
        error_label="SRT fix",
        done_message="Done! Fixed SRT saved next to the original as *_fixed.srt.",
        next_message="Next: drag the *_fixed.srt file onto 4_SplitSRT.bat",
    )


def make_split_srt_bat(script_path: Path) -> str:
    return _make_srt_step_bat(
        script_path,
        missing_message="Drag your fixed .srt file onto this script.",
        progress_message="Splitting SRT into transcript_part files",
        command="--split-srt",
        error_label="SRT split",
        done_message="Done! transcript_part files are ready in the folder.",
        next_message="Double-click 5_AnalyzeHighlights.bat or 6_RunAllSteps.bat to find highlights.",
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


ANALYSIS_MINI_STAGES = BIG_STEP_MINI_STAGES[4]


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
    if kind == "mic_wav":
        exact_path = target_folder / f"{base_name}_mic.wav"
        candidates = target_folder.glob("*_mic.wav")
        description = f"{exact_path} or newest *_mic.wav"
    elif kind == "prepared_chunks":
        exact_path = target_folder / f"{base_name}_mic_transcription_chunks"
        manifest_path = exact_path / "chunk_manifest.json"
        mic_path = target_folder / f"{base_name}_mic.wav"
        description = f"{mic_path} plus saved chunks in {exact_path}"
        if mic_path.is_file() and manifest_path.is_file():
            try:
                _load_prepared_audio_chunks(mic_path)
            except (OSError, RuntimeError, ValueError, TypeError):
                pass
            else:
                return exact_path, description
        return None, description
    elif kind == "raw_srt":
        exact_path = target_folder / f"{base_name}_mic.srt"
        candidates = (
            path for path in target_folder.glob("*.srt")
            if not path.stem.casefold().endswith("_fixed")
        )
        description = f"{exact_path} or newest non-fixed *.srt"
    elif kind == "fixed_srt":
        exact_path = target_folder / f"{base_name}_mic_fixed.srt"
        candidates = target_folder.glob("*_fixed.srt")
        description = f"{exact_path} or newest *_fixed.srt"
    elif kind == "transcript_part":
        exact_path = target_folder / "transcript_part1.txt"
        candidates = target_folder.glob("transcript_part*.txt")
        description = f"{exact_path} or newest transcript_part*.txt"
    elif kind == "highlights_csv":
        exact_path = None
        candidates = target_folder.glob("top*_highlights.csv")
        description = f"{target_folder} / top*_highlights.csv"
    else:
        raise ValueError(f"Unknown run-all file kind: {kind}")

    path = exact_path if exact_path is not None and exact_path.exists() else newest_file(candidates)
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

def run_all_gui(target_folder: Path, base_name: str) -> int:
    """Run the five pipeline steps and record each attempt's wall time.

    Successful, skipped, failed, and stopped attempts are appended to
    ``View_Pipeline_Duration_History.csv`` beside this script. The duration viewer
    displays averages from successful attempts.
    """
    target_folder = target_folder.resolve()
    steps = build_run_all_steps(target_folder, base_name)

    import tkinter as tk
    from tkinter import messagebox, ttk

    try:
        from PIL import Image, ImageTk
    except ImportError:
        Image = None
        ImageTk = None

    events: queue.Queue[tuple[str, object]] = queue.Queue()
    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    run_log_path = make_step6_log_path(target_folder)
    run_log = run_log_path.open("w", encoding="utf-8", buffering=1)
    run_log_closed = {"value": False}
    run_log.write("Step 6 run log\n")
    run_log.write(f"Started: {datetime.now().isoformat(timespec='seconds')}\n")
    run_log.write(f"Run ID: {run_id}\n")
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
    root.title("Run All  /  VOD Highlight Pipeline")
    root.geometry("1440x900")
    root.minsize(1120, 720)
    root.configure(bg="#101419")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(
        ".",
        background="#101419",
        foreground="#f2f2f2",
        fieldbackground="#1e252e",
        bordercolor="#33404d",
    )
    style.configure("TFrame", background="#101419")
    style.configure("TLabel", background="#101419", foreground="#f2f2f2")
    style.configure(
        "TButton",
        background="#26313c",
        foreground="#f2f2f2",
        bordercolor="#425263",
        padding=(10, 6),
        font=("Segoe UI", 9, "bold"),
    )
    style.map("TButton", background=[("active", "#344352"), ("disabled", "#202832")])
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

    status_var = tk.StringVar(value="READY  •  Preparing the five big steps")
    progress_var = tk.DoubleVar(value=0)
    exit_code = {"value": 0}
    status_colors = {
        "Waiting": "#8c9aaa",
        "Running": "#ffd166",
        "Done": "#7CFC98",
        "Skipped": "#7CFC98",
        "Failed": "#ff6b6b",
        "Stopped": "#ffb86c",
    }
    status_card_colors = {
        "Waiting": "#1b2027",
        "Running": "#302817",
        "Done": "#17271d",
        "Skipped": "#17271d",
        "Failed": "#321b20",
        "Stopped": "#302319",
    }
    step_name_labels: list[tk.Label] = []
    step_status_labels: list[tk.Label] = []
    step_card_frames: list[tk.Frame] = []
    step_detail_vars: list[tk.StringVar] = []
    step_detail_labels: list[tk.Label] = []
    step_runtime: list[dict[str, str]] = []
    mini_stage_status_vars: dict[str, tk.StringVar] = {}
    mini_stage_status_labels: dict[str, tk.Label] = {}
    mini_stage_detail_vars: dict[str, tk.StringVar] = {}
    mini_stage_card_frames: list[tk.Frame] = []
    summary_status_label: dict[str, tk.Label | None] = {"widget": None}
    console_line_count = {"value": 0}


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

    def render_step_detail(index: int) -> None:
        state = step_runtime[index]
        detail_parts = []
        if state["stage"]:
            detail_parts.append(f"Mini-process: {state['stage']}")
        if state["task"]:
            detail_parts.append(f"Current: {state['task']}")
        if state["model"]:
            detail_parts.append(state["model"])
        if state["message"]:
            detail_parts.append(state["message"])
        if state["next"]:
            detail_parts.append(state["next"])
        step_detail_vars[index].set("\n".join(detail_parts))

    def render_mini_stages(index: int) -> None:
        stages = _mini_stages_for_step(index, target_folder)
        mini_frame.configure(text=f"  STEP {index + 1}  /  MINI-PROCESSES  ")
        for card in mini_stage_card_frames:
            card.destroy()
        mini_stage_card_frames.clear()
        mini_stage_status_vars.clear()
        mini_stage_status_labels.clear()
        mini_stage_detail_vars.clear()

        for column, (stage_code, stage_name) in enumerate(stages):
            mini_frame.columnconfigure(column, weight=1)
            mini_card = tk.Frame(mini_frame, bg="#1b2027", padx=7, pady=7)
            mini_card.grid(
                row=0,
                column=column,
                sticky="ew",
                padx=(0 if column == 0 else 3, 0),
            )
            mini_stage_card_frames.append(mini_card)
            tk.Label(
                mini_card,
                text=stage_code,
                bg="#1b2027",
                fg="#7f8b99",
                font=("Consolas", 9, "bold"),
            ).pack(anchor="w")
            tk.Label(
                mini_card,
                text=stage_name,
                bg="#1b2027",
                fg="#d9e2ec",
                font=("Segoe UI", 9, "bold"),
            ).pack(anchor="w", pady=(2, 4))
            status_var = tk.StringVar(value="WAITING")
            mini_stage_status_vars[stage_code] = status_var
            status_label = tk.Label(
                mini_card,
                textvariable=status_var,
                bg="#1b2027",
                fg=status_colors["Waiting"],
                font=("Segoe UI", 8, "bold"),
            )
            status_label.pack(anchor="w")
            mini_stage_status_labels[stage_code] = status_label
            detail_var = tk.StringVar(value=f"Waiting for Step {index + 1}")
            mini_stage_detail_vars[stage_code] = detail_var
            tk.Label(
                mini_card,
                textvariable=detail_var,
                bg="#1b2027",
                fg="#8795a5",
                anchor="w",
                justify="left",
                wraplength=125,
                font=("Segoe UI", 8),
            ).pack(anchor="w", fill="x", pady=(3, 0))

    def set_mini_stage(code: str, status: str, detail: str) -> None:
        status_var = mini_stage_status_vars.get(code)
        status_label = mini_stage_status_labels.get(code)
        detail_var = mini_stage_detail_vars.get(code)
        if status_var is None or status_label is None or detail_var is None:
            return
        status_var.set(status.upper())
        detail_var.set(detail)
        status_label.configure(fg=status_colors.get(status, status_colors["Waiting"]))

    def set_step_mini(index: int, code: str, status: str, detail: str) -> None:
        stage_names = dict(_mini_stages_for_step(index, target_folder))
        stage_name = stage_names.get(code)
        if stage_name is None:
            return
        step_runtime[index]["stage"] = f"{code}. {stage_name}"
        set_mini_stage(code, status, detail)
        render_step_detail(index)

    def set_all_step_minis(index: int, status: str, detail: str) -> None:
        for code, _name in _mini_stages_for_step(index, target_folder):
            set_step_mini(index, code, status, detail)

    def begin_step_minis(index: int) -> None:
        render_mini_stages(index)
        first_code, _name = _mini_stages_for_step(index, target_folder)[0]
        set_step_mini(index, first_code, "Running", "Starting mini-process")

    def set_step_status(index: int, value: str) -> None:
        step_status_vars[index].set(value.upper())
        if value == "Running":
            begin_step_minis(index)
        elif value in {"Done", "Skipped"}:
            set_all_step_minis(index, value, "Big-step output is ready")
            step_runtime[index]["next"] = ""
        color = status_colors.get(value, status_colors["Waiting"])
        card_color = status_card_colors.get(value, status_card_colors["Waiting"])
        for widget in (
            step_card_frames[index],
            step_name_labels[index],
            step_status_labels[index],
            step_detail_labels[index],
        ):
            widget.configure(bg=card_color)
        step_name_labels[index].configure(fg=color)
        step_status_labels[index].configure(fg=color)
        step_detail_labels[index].configure(
            fg="#d9e2ec" if value not in {"Failed", "Stopped"} else color
        )
        if summary_status_label["widget"] is not None:
            summary_status_label["widget"].configure(fg=color)
        render_step_detail(index)

    def update_step_detail(index: int, message: str) -> None:
        step_runtime[index]["message"] = message
        render_step_detail(index)

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
                    set_mini_stage(code, "Skipped", "Checkpoint reused")
                else:
                    state["task"] = f"Running mini-process: {stage_info['task']}"
                    state["message"] = "Reading live analyzer output..."
                    set_mini_stage(code, "Running", "Live output received")
                state["model"] = stage_info["model"]
                state["next"] = stage_info["next"]
                break

            if "stage finished in " in lowered and state["stage"]:
                code, _name = ANALYSIS_STAGE_BY_LABEL[state["stage"].casefold()]
                set_mini_stage(code, "Done", line[:74])
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

    main_frame = tk.Frame(root, bg="#101419", padx=18, pady=16)
    main_frame.pack(fill="both", expand=True)
    main_frame.columnconfigure(0, weight=6)
    main_frame.columnconfigure(1, weight=4)
    main_frame.rowconfigure(3, weight=1)

    header_frame = tk.Frame(main_frame, bg="#101419")
    header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
    header_frame.columnconfigure(0, weight=1)
    tk.Label(
        header_frame,
        text="RUN ALL  /  VOD HIGHLIGHT PIPELINE",
        anchor="w",
        bg="#101419",
        fg="#f4f7fb",
        font=("Segoe UI", 20, "bold"),
    ).grid(row=0, column=0, sticky="w")
    tk.Label(
        header_frame,
        text=f"{base_name}  •  {target_folder}",
        anchor="w",
        bg="#101419",
        fg="#91a0b2",
        font=("Segoe UI", 9),
    ).grid(row=1, column=0, sticky="w", pady=(3, 0))
    tk.Label(
        header_frame,
        text="LIVE / CHECKPOINTED / RESUMABLE",
        anchor="e",
        bg="#101419",
        fg="#5eead4",
        font=("Segoe UI", 9, "bold"),
    ).grid(row=0, column=1, rowspan=2, sticky="e")

    status_row = tk.Frame(main_frame, bg="#182029", padx=12, pady=10)
    status_row.grid(row=1, column=0, sticky="ew", padx=(0, 12), pady=(0, 10))
    status_row.columnconfigure(0, weight=1)
    summary_label = tk.Label(
        status_row,
        textvariable=status_var,
        anchor="w",
        justify="left",
        bg="#182029",
        fg="#f4f7fb",
        font=("Segoe UI", 11, "bold"),
    )
    summary_label.grid(row=0, column=0, sticky="ew")
    summary_status_label["widget"] = summary_label
    progress = ttk.Progressbar(
        status_row,
        variable=progress_var,
        maximum=len(steps),
        style="Orange.Horizontal.TProgressbar",
        length=220,
    )
    progress.grid(row=0, column=1, sticky="ew", padx=(16, 12))
    stop_button = ttk.Button(status_row, text="STOP RUN", style="Stop.TButton")
    stop_button.grid(row=0, column=2, sticky="e")

    left_frame = tk.Frame(main_frame, bg="#101419")
    left_frame.grid(row=2, column=0, rowspan=2, sticky="nsew", padx=(0, 12))
    left_frame.columnconfigure(0, weight=1)
    left_frame.rowconfigure(0, weight=1)
    left_frame.rowconfigure(2, weight=1)

    step_frame = tk.LabelFrame(
        left_frame,
        text="  BIG STEPS  ",
        bg="#101419",
        fg="#d9e2ec",
        bd=1,
        relief="groove",
        padx=10,
        pady=10,
        font=("Segoe UI", 10, "bold"),
    )
    step_frame.grid(row=0, column=0, sticky="nsew")
    step_frame.columnconfigure(0, weight=1)
    step_status_vars: list[tk.StringVar] = []
    for row, step in enumerate(steps):
        card_color = status_card_colors["Waiting"]
        card = tk.Frame(step_frame, bg=card_color, padx=10, pady=7)
        card.grid(row=row, column=0, sticky="ew", pady=(0 if row == 0 else 6, 0))
        card.columnconfigure(1, weight=1)
        step_card_frames.append(card)

        accent = tk.Frame(card, bg="#46515e", width=5)
        accent.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 10))

        name_label = tk.Label(
            card,
            text=step.label,
            anchor="w",
            bg=card_color,
            fg="#f4f7fb",
            font=("Segoe UI", 11, "bold"),
        )
        name_label.grid(row=0, column=1, sticky="ew")
        step_name_labels.append(name_label)

        state_var = tk.StringVar(value="WAITING")
        step_status_vars.append(state_var)
        status_label = tk.Label(
            card,
            textvariable=state_var,
            width=10,
            anchor="e",
            bg=card_color,
            fg=status_colors["Waiting"],
            font=("Segoe UI", 9, "bold"),
        )
        status_label.grid(row=0, column=2, rowspan=2, sticky="ne", padx=(10, 0))
        step_status_labels.append(status_label)

        detail_var = tk.StringVar()
        step_detail_vars.append(detail_var)
        detail_label = tk.Label(
            card,
            textvariable=detail_var,
            anchor="w",
            justify="left",
            wraplength=760,
            bg=card_color,
            fg="#d9e2ec",
            font=("Segoe UI", 9),
        )
        detail_label.grid(row=1, column=1, sticky="ew", pady=(3, 0))
        step_detail_labels.append(detail_label)

    mini_frame = tk.LabelFrame(
        left_frame,
        text="  STEP 5  /  MINI-PROCESSES  ",
        bg="#101419",
        fg="#d9e2ec",
        bd=1,
        relief="groove",
        padx=8,
        pady=8,
        font=("Segoe UI", 10, "bold"),
    )
    mini_frame.grid(row=1, column=0, sticky="ew", pady=(10, 10))
    render_mini_stages(0)

    console_frame = tk.LabelFrame(
        left_frame,
        text="  LIVE CONSOLE  ",
        bg="#101419",
        fg="#d9e2ec",
        bd=1,
        relief="groove",
        padx=8,
        pady=8,
        font=("Segoe UI", 10, "bold"),
    )
    console_frame.grid(row=2, column=0, sticky="nsew")
    console_frame.rowconfigure(0, weight=1)
    console_frame.columnconfigure(0, weight=1)
    log_box = tk.Text(
        console_frame,
        height=12,
        wrap="word",
        state="disabled",
        bg="#0b0f14",
        fg="#e7edf4",
        insertbackground="#e7edf4",
        selectbackground="#334155",
        relief="flat",
        padx=10,
        pady=8,
        font=("Consolas", 9),
    )
    log_box.tag_configure("info", foreground="#e7edf4")
    log_box.tag_configure("success", foreground="#7CFC98")
    log_box.tag_configure("failure", foreground="#ff6b6b")
    log_box.tag_configure("warning", foreground="#ffd166")
    log_box.grid(row=0, column=0, sticky="nsew")
    log_scrollbar = ttk.Scrollbar(console_frame, orient="vertical", command=log_box.yview)
    log_scrollbar.grid(row=0, column=1, sticky="ns")
    log_box.configure(yscrollcommand=log_scrollbar.set)

    gallery_frame = tk.LabelFrame(
        main_frame,
        text="  BEST OF GALLERY  ",
        bg="#101419",
        fg="#d9e2ec",
        bd=1,
        relief="groove",
        padx=10,
        pady=10,
        font=("Segoe UI", 10, "bold"),
    )
    gallery_frame.grid(row=1, column=1, rowspan=3, sticky="nsew")
    gallery_frame.rowconfigure(0, weight=1)
    gallery_frame.columnconfigure(0, weight=1)
    gallery_frame.columnconfigure(1, weight=1)

    image_label = tk.Label(
        gallery_frame,
        text="Loading gallery images...",
        anchor="center",
        justify="center",
        bg="#0b0f14",
        fg="#9aa8b8",
        font=("Segoe UI", 10),
    )
    image_label.grid(row=0, column=0, columnspan=2, sticky="nsew")

    gallery_paths = gallery_image_paths()
    gallery_index = {"value": 0}
    gallery_photo = {"value": None}
    current_gallery_path = {"value": None}
    gallery_render_after = {"id": None}
    gallery_rotation_after = {"id": None}
    gallery_meta_var = tk.StringVar(value="")

    tk.Label(
        gallery_frame,
        textvariable=gallery_meta_var,
        anchor="w",
        bg="#101419",
        fg="#9aa8b8",
        font=("Segoe UI", 9),
    ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 4))

    def append_log(text: str, tag: str | None = None) -> None:
        console_line_count["value"] += text.count("\n") or 1
        if console_line_count["value"] == 1 or console_line_count["value"] % 25 == 0:
            console_frame.configure(text=f"  LIVE CONSOLE  •  {console_line_count['value']:,} lines  ")
        log_box.configure(state="normal")
        log_box.insert("end", text, tag or _console_log_tag(text))
        log_box.see("end")
        log_box.configure(state="disabled")
        write_run_log(text)

    def render_gallery_image(image_path: Path) -> None:
        try:
            available_width = max(image_label.winfo_width() - 20, 240)
            available_height = max(image_label.winfo_height() - 20, 240)
            if available_width <= 240 or available_height <= 240:
                available_width = max(gallery_frame.winfo_width() - 24, 320)
                available_height = max(gallery_frame.winfo_height() - 72, 320)
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
            else:
                gallery_photo["value"] = tk.PhotoImage(file=str(image_path))
            image_label.configure(image=gallery_photo["value"], text="")
        except Exception as exc:
            image_label.configure(text=f"Could not load gallery image:\n{exc}", image="")
            gallery_photo["value"] = None

    def show_gallery_image(offset: int = 0) -> None:
        if not gallery_paths:
            image_label.configure(text=f"No images found in:\n{GALLERY_DIR}", image="")
            gallery_meta_var.set("Gallery folder is empty or unavailable")
            return

        gallery_index["value"] = (gallery_index["value"] + offset) % len(gallery_paths)
        current_path = gallery_paths[gallery_index["value"]]
        current_gallery_path["value"] = current_path
        gallery_meta_var.set(
            f"{gallery_index['value'] + 1} / {len(gallery_paths)}  •  {current_path.name}"
        )
        render_gallery_image(current_path)

    ttk.Button(gallery_frame, text="‹  Previous", command=lambda: show_gallery_image(-1)).grid(
        row=2, column=0, sticky="ew", pady=(4, 0), padx=(0, 4)
    )
    ttk.Button(gallery_frame, text="Next  ›", command=lambda: show_gallery_image(1)).grid(
        row=2, column=1, sticky="ew", pady=(4, 0), padx=(4, 0)
    )

    def rerender_gallery_image(_event: object | None = None) -> None:
        if current_gallery_path["value"] is None:
            return
        if gallery_render_after["id"] is not None:
            root.after_cancel(gallery_render_after["id"])
        gallery_render_after["id"] = root.after(
            120,
            lambda: render_gallery_image(current_gallery_path["value"]),
        )

    image_label.bind("<Configure>", rerender_gallery_image)

    def rotate_gallery() -> None:
        show_gallery_image(1)
        gallery_rotation_after["id"] = root.after(8000, rotate_gallery)

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
            ollama_base_url = _ollama_base_url(OLLAMA_URL)
            if not ollama_is_reachable(ollama_base_url):
                message = ollama_not_ready_message(ollama_base_url)
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
        """Put Windows into sleep after a successful, opted-in pipeline run."""
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
                status_var.set(f"{steps[index].label}: {value}")
                append_log(
                    f"[STATUS] {steps[index].label}: {value}\n",
                    "success" if value in {"Done", "Skipped"} else _console_log_tag(value),
                )
            elif kind == "detail":
                index, message = payload
                update_step_detail(index, str(message))
            elif kind == "ollama_unavailable":
                messagebox.showwarning("Ollama is not running", str(payload), parent=root)
            elif kind == "progress":
                progress_var.set(payload)
            elif kind == "output":
                index, line = payload
                append_log(line)
                update_step_from_output(index, line)
            elif kind == "log":
                append_log(payload)
            elif kind == "failed":
                exit_code["value"] = 1
                status_var.set(str(payload))
                summary_label.configure(fg=status_colors["Failed"])
                append_log(f"\nERROR: {payload}\n", "failure")
                stop_button.configure(state="disabled")
            elif kind == "stopped":
                last_done = payload
                status_var.set(
                    f"Stopped by user. Last fully completed step: {last_done}. "
                    f"Run 6_RunAllSteps.bat again to continue from there."
                )
                summary_label.configure(fg=status_colors["Stopped"])
                append_log(
                    f"\nStopped by user. Last fully completed step: {last_done}.\n"
                    "Ollama models unloaded. Run 6_RunAllSteps.bat again to continue.\n",
                    "warning",
                )
                stop_button.configure(state="disabled", text="Stopped")
            elif kind == "complete":
                status_message = str(payload)
                if AUTO_SLEEP_AFTER_PIPELINE:
                    status_message += " PC will go to sleep now."
                status_var.set(status_message)
                summary_label.configure(fg=status_colors["Done"])
                append_log(f"\n{status_message}\n", "success")
                progress_var.set(len(steps))
                write_run_log(f"[PROGRESS] {len(steps)}/{len(steps)}\n")
                stop_button.configure(state="disabled")
                if AUTO_SLEEP_AFTER_PIPELINE:
                    append_log("[sleep] Auto-sleep is enabled; suspending Windows.\n", "success")
                    threading.Thread(target=request_system_sleep, daemon=True).start()

        root.after(80, drain_events)

    show_gallery_image()
    if len(gallery_paths) > 1:
        gallery_rotation_after["id"] = root.after(8000, rotate_gallery)

    def on_window_close() -> None:
        if gallery_render_after["id"] is not None:
            root.after_cancel(gallery_render_after["id"])
        if gallery_rotation_after["id"] is not None:
            root.after_cancel(gallery_rotation_after["id"])
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
        "2_TranscribeAudio.bat": make_transcribe_bat(script_path),
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
        print("   1_ExtractMicAudio.bat   <- isolate the voice, then split/save all chunks")
        print("                              (single-track Twitch-style VOD)")
    else:
        print("   1_ExtractMicAudio.bat   <- extract mic track, then split/save all chunks")
        print("                              (separate-track local recording)")
    print("   2_TranscribeAudio.bat   <- run Whisper on each saved chunk, then stitch the SRT")
    print("   3_FixSRT.bat            <- drag stitched .srt onto this to fix repeats/timestamps")
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
            fix_srt(Path(args.fix_srt))
            return 0

        if args.split_srt:
            split_srt_into_chunks(Path(args.split_srt), args.chunk_minutes)
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