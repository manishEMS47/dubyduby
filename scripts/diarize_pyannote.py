#!/usr/bin/env python3
"""Run pyannote.audio speaker diarization and reassign speaker labels on tokens.json.

Soniox's built-in diarization is often inaccurate for same-gender speakers
(50%+ ABA toggle on the Lex×Jensen interview). pyannote.audio is the industry-
standard diarization tool and produces 5-10% error rate even on similar-pitch
voices.

Workflow:
  1. Load original audio.mp3
  2. Run pyannote/speaker-diarization-3.1 → segments (start, end, speaker_id)
  3. For each Soniox token, find the pyannote segment its midpoint falls in
     and overwrite token['speaker']
  4. Backup original tokens.json → tokens.soniox.json
  5. Write new tokens.json + diarization.json (raw pyannote output)

Reads:
  output/<video_id>/1_source/audio.mp3
  output/<video_id>/2_transcript/tokens.json
  $HF_TOKEN (from env or ~/.config/secrets/huggingface.env)

Writes:
  output/<video_id>/2_transcript/tokens.json (speakers reassigned)
  output/<video_id>/2_transcript/tokens.soniox.json (backup)
  output/<video_id>/2_transcript/diarization.json (raw segments)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from pyannote.audio import Pipeline


def to_pyannote_wav(audio_path: Path) -> Path:
    """Convert any input audio to 16kHz mono WAV. pyannote chokes on mp3 files
    whose decoded sample count doesn't match the expected duration (rounding
    issues in mp3 frame boundaries), so we re-encode to a clean PCM WAV first.
    """
    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(audio_path),
        "-ac", "1", "-ar", "16000",
        str(tmp),
    ], check=True)
    return tmp


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: diarize_pyannote.py <video_id>")
    video_id = sys.argv[1]
    base = Path(f"output/{video_id}")
    audio = base / "1_source" / "audio.mp3"
    tokens_path = base / "2_transcript" / "tokens.json"

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        sys.exit("HF_TOKEN missing — source ~/.config/secrets/huggingface.env first")

    print(f"[diarize] loading pyannote pipeline (first run ~30s, cached after)...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    # Move to MPS (Apple Silicon GPU) if available for ~5-10x speedup
    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
        print("[diarize] using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print("[diarize] using CUDA GPU")
    else:
        print("[diarize] using CPU")

    print(f"[diarize] converting audio to 16kHz mono WAV (clean for pyannote)...")
    wav_path = to_pyannote_wav(audio)
    print(f"[diarize] running diarization on {wav_path.name}...")
    try:
        diarization = pipeline(str(wav_path))
    finally:
        wav_path.unlink(missing_ok=True)

    # pyannote 4.x returns DiarizeOutput with two views:
    #   - speaker_diarization: includes overlapping turns
    #   - exclusive_speaker_diarization: no overlaps (one speaker per moment)
    # For mapping transcript tokens we need the exclusive view.
    annotation = diarization.exclusive_speaker_diarization
    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append((int(turn.start * 1000), int(turn.end * 1000), speaker))
    segments.sort(key=lambda s: s[0])

    speaker_ids = sorted({s[2] for s in segments})
    print(f"[diarize] {len(segments)} segments, {len(speaker_ids)} unique speakers: {speaker_ids}")

    # Map pyannote labels (e.g. SPEAKER_00) → numeric ids ("1", "2", ...) to
    # match the rest of the dubyduby pipeline (speakers.json keys).
    label_to_id = {label: str(i + 1) for i, label in enumerate(speaker_ids)}

    # Save raw diarization
    diar_path = base / "2_transcript" / "diarization.json"
    diar_path.write_text(json.dumps(
        [{"start_ms": s, "end_ms": e, "speaker": label_to_id[lbl], "pyannote_label": lbl}
         for s, e, lbl in segments],
        ensure_ascii=False, indent=2,
    ))

    # Backup original tokens
    tokens_data = json.loads(tokens_path.read_text())
    backup_path = base / "2_transcript" / "tokens.soniox.json"
    if not backup_path.exists():
        shutil.copy(tokens_path, backup_path)
        print(f"[diarize] backed up Soniox tokens → {backup_path.name}")

    # Reassign each token's speaker based on midpoint overlap with pyannote segments.
    # Use bisect for fast lookup since segments are sorted.
    import bisect
    seg_starts = [s[0] for s in segments]

    def find_speaker(mid_ms: int) -> str:
        """Locate the segment whose [start, end) contains mid_ms.
        If the token sits in a tiny gap between two segments, fall back to
        the closer one — pyannote leaves small gaps (~100-500ms) between
        adjacent turns that we should still attribute to one of them.
        """
        idx = bisect.bisect_right(seg_starts, mid_ms) - 1
        if 0 <= idx < len(segments):
            s, e, lbl = segments[idx]
            if s <= mid_ms < e:
                return label_to_id[lbl]
        # Gap fallback: pick whichever neighboring segment is closer
        candidates = []
        if 0 <= idx < len(segments):
            s, e, lbl = segments[idx]
            candidates.append((min(abs(mid_ms - s), abs(mid_ms - e)), label_to_id[lbl]))
        if 0 <= idx + 1 < len(segments):
            s, e, lbl = segments[idx + 1]
            candidates.append((min(abs(mid_ms - s), abs(mid_ms - e)), label_to_id[lbl]))
        if candidates:
            return min(candidates)[1]
        # No segments at all (shouldn't happen) — keep original
        return None

    reassigned = 0
    unmapped = 0
    for t in tokens_data["tokens"]:
        mid_ms = (t["start_ms"] + t["end_ms"]) // 2
        new_spk = find_speaker(mid_ms)
        if new_spk:
            if t.get("speaker") != new_spk:
                reassigned += 1
            t["speaker"] = new_spk
        else:
            unmapped += 1

    tokens_path.write_text(json.dumps(tokens_data, ensure_ascii=False, indent=2))
    print(f"[diarize] reassigned {reassigned} tokens; {unmapped} unmapped (no segment)")
    print(f"[diarize] OK → {tokens_path}")


if __name__ == "__main__":
    main()
