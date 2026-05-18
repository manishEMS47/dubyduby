#!/usr/bin/env python3
"""Place per-utterance wavs at source video timeline positions.

Each utt anchors to utterance.start_ms (original English speaker mouth-cue).
If KO synth duration exceeds the slot before next utt, apply per-utt atempo
to fit (clamped [1.0, 1.6]). Silence in source video → silence in dub.

Reads:
  output/<video_id>/3_translation/utterances.json
  output/<video_id>/4_synth/utt-NNN.wav

Writes:
  output/<video_id>/6_final/dubbed_audio.wav (placed timeline)
  output/<video_id>/5_intermediate/placement.json (per-utt final timing)
"""
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np


ATEMPO_MIN = 1.0
ATEMPO_MAX = 1.6  # hard limit — beyond this Korean sounds rushed

# Adaptive pause shrink: Korean synth is shorter than English source (filler-free,
# higher info density) so anchoring at original utt.start_ms leaves big silences.
# Solution: if the gap between prev utt's actual end and this utt's anchor exceeds
# MIN_NATURAL_PAUSE_MS, keep only SHRINK_RATIO of it (capped at MAX_PAUSE_MS).
# Reactive per-video: short-pause videos barely change; long-pause videos (heavy
# filler speakers) shrink the most. Subtitle timing follows actual placement.
MIN_NATURAL_PAUSE_MS = 150   # natural breath, always preserved
MAX_PAUSE_MS = 1500          # absolute upper bound on inter-utt silence
SHRINK_RATIO = 0.4           # keep 40% of the raw silence

# Hard cap on cumulative drift between KO playback and source video timeline.
# Without this, silence shrinking compounds — by 25min the KO subtitle ran
# 47s ahead of the speaker's mouth. With cap=2s, dub stays within 2s of source
# at all times; long natural pauses absorb the catch-up.
MAX_CUMULATIVE_DRIFT_MS = 2000


def read_wav_i16(path: Path):
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16), sr


def write_wav_i16(path: Path, samples: np.ndarray, sr: int):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.astype(np.int16).tobytes())


def atempo_wav(in_path: Path, out_path: Path, factor: float):
    """Apply ffmpeg atempo filter."""
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(in_path),
        "-af", f"atempo={factor:.4f}",
        str(out_path),
    ], check=True)


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: place_timeline.py <video_id> <video_duration_ms>")
    video_id = sys.argv[1]
    video_duration_ms = int(sys.argv[2])
    base = Path(f"output/{video_id}")
    utts = json.load(open(base / "3_translation" / "utterances.json"))
    synth_dir = base / "4_synth"
    inter_dir = base / "5_intermediate"
    final_dir = base / "6_final"
    inter_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    # Load all utt wavs first to get sample rate
    first_wav = synth_dir / "utt-000.wav"
    _, sr = read_wav_i16(first_wav)
    total_samples = int(video_duration_ms * sr / 1000)
    buf = np.zeros(total_samples, dtype=np.int32)  # accumulator (int32 for safe mixing)

    placements = []
    prev_actual_end_ms = 0  # tracks where the previous utt's audio actually ended
    n = len(utts)
    for i, u in enumerate(utts):
        wav_path = synth_dir / f"utt-{i:03d}.wav"
        if not wav_path.exists():
            continue
        samples, _ = read_wav_i16(wav_path)
        wav_dur_ms = int(len(samples) / sr * 1000)
        anchor_start_ms = u["start_ms"]

        # Adaptive shift: if anchor leaves a big silent gap after the previous
        # utt's actual end, shrink it. Otherwise (gap already small/natural)
        # keep the anchor.
        raw_gap_ms = anchor_start_ms - prev_actual_end_ms
        if raw_gap_ms > MIN_NATURAL_PAUSE_MS:
            shrunk_gap_ms = max(
                int(raw_gap_ms * SHRINK_RATIO),
                MIN_NATURAL_PAUSE_MS,
            )
            shrunk_gap_ms = min(shrunk_gap_ms, MAX_PAUSE_MS)
            start_ms = prev_actual_end_ms + shrunk_gap_ms
        else:
            start_ms = anchor_start_ms

        # Cap cumulative drift: KO playback can lead the source video by at
        # most MAX_CUMULATIVE_DRIFT_MS. Otherwise silence-shrinking compounds
        # and the dub ends up tens of seconds ahead of the speaker. If we'd
        # drift past the cap, pin start to (anchor - cap) — long natural
        # pauses then absorb the catch-up gracefully.
        min_allowed_start = anchor_start_ms - MAX_CUMULATIVE_DRIFT_MS
        if start_ms < min_allowed_start:
            start_ms = min_allowed_start

        # slot is the time available before the next utt's anchor — we still
        # respect the source video pacing for atempo decisions
        if i + 1 < n:
            next_anchor_ms = utts[i + 1]["start_ms"]
        else:
            next_anchor_ms = video_duration_ms
        slot_ms = next_anchor_ms - start_ms

        atempo = 1.0
        if wav_dur_ms > slot_ms and slot_ms > 0:
            atempo = min(wav_dur_ms / slot_ms, ATEMPO_MAX)
        if atempo > 1.0:
            scaled_path = inter_dir / f"utt-{i:03d}_scaled.wav"
            atempo_wav(wav_path, scaled_path, atempo)
            samples, _ = read_wav_i16(scaled_path)
            scaled_path.unlink()

        start_sample = int(start_ms * sr / 1000)
        end_sample = min(start_sample + len(samples), total_samples)
        copy_len = end_sample - start_sample
        buf[start_sample:end_sample] += samples[:copy_len].astype(np.int32)

        actual_end_ms = start_ms + int(copy_len / sr * 1000)
        prev_actual_end_ms = actual_end_ms

        placements.append({
            "i": i,
            "start_ms": start_ms,
            "end_ms": actual_end_ms,
            "atempo": round(atempo, 4),
            "slot_ms": slot_ms,
            "raw_dur_ms": wav_dur_ms,
            "anchor_start_ms": anchor_start_ms,
            "shift_ms": start_ms - anchor_start_ms,
        })

    # Clip int32 back to int16 range (no clipping expected since utts don't overlap)
    buf = np.clip(buf, -32768, 32767)
    write_wav_i16(final_dir / "dubbed_audio.wav", buf, sr)

    json.dump(placements, open(inter_dir / "placement.json", "w"), indent=2)
    n_scaled = sum(1 for p in placements if p["atempo"] > 1.0)
    n_shifted = sum(1 for p in placements if p["shift_ms"] < 0)
    total_shift_ms = -sum(p["shift_ms"] for p in placements if p["shift_ms"] < 0)
    print(f"[place] {len(placements)} utts placed at adaptive timeline")
    print(f"[place] {n_scaled} scaled (atempo > 1.0), {len(placements) - n_scaled} natural")
    print(f"[place] {n_shifted} shifted earlier (silence cut), total shrink {total_shift_ms/1000:.0f}s")


if __name__ == "__main__":
    main()
