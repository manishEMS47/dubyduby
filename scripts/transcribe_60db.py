#!/usr/bin/env python3
"""60db STT batch — drop-in alternative to transcribe.sh (Soniox).

Produces the SAME files in the SAME schema as the Soniox path so every
downstream script (match_timing.py, analyze_speakers.py, diarize_pyannote.py)
works unchanged:

  output/<video_id>/2_transcript/tokens.json
    {"tokens": [{"text", "start_ms", "end_ms", "speaker"}, ...]}
  output/<video_id>/2_transcript/transcript.md

60db's native /stt response differs from Soniox and is adapted here:
  - times are SECONDS (float)        → start_ms/end_ms (int, ×1000)
  - word units in segments[].words[] → flat token stream (space-prefixed)
  - speaker labels "SPEAKER_00"...    → "1", "2", ... (pyannote convention,
                                        matches diarize_pyannote.py output)

Selected via DUBYDUBY_STT=60db in dub.sh. Soniox remains the default.

Reads:
  output/<video_id>/1_source/audio.mp3
  SIXTYDB_API_KEY        (from env or .env at repo root)
  SIXTYDB_BASE_URL       (optional, default https://api.60db.ai)
  DUBYDUBY_STT_LANG      (optional, default "en"; "auto" for auto-detect)

Usage: transcribe_60db.py <video_id>
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# 60db /stt hard limits (https://docs.60db.ai/api-reference/stt/speech-to-text)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def load_env(repo_root: Path) -> None:
    """Populate os.environ from .env (does not override real env vars).
    Mirrors the `set -a && . ./.env` that transcribe.sh does for Soniox.
    """
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def fit_under_limit(audio: Path) -> Path:
    """60db caps uploads at 10 MB. yt-dlp mp3 of a ~30-min video is larger, so
    re-encode to mono 64 kbps mp3 when needed. Returns a path to an upload-ready
    file (may be a temp file the caller should delete). Warns if still too big.
    """
    if audio.stat().st_size <= MAX_UPLOAD_BYTES:
        return audio
    print(f"[60db] audio {audio.stat().st_size/1e6:.1f}MB > 10MB limit — "
          f"re-encoding to mono 64kbps mp3", file=sys.stderr)
    tmp = Path(tempfile.mkstemp(suffix=".mp3")[1])
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(audio), "-ac", "1", "-b:a", "64k", str(tmp),
    ], check=True)
    if tmp.stat().st_size > MAX_UPLOAD_BYTES:
        print(f"[60db] WARNING: still {tmp.stat().st_size/1e6:.1f}MB after "
              f"re-encode. 60db may reject it. Cut the video with "
              f"`dub.sh <URL> <seconds>` or split long sources.", file=sys.stderr)
    return tmp


def call_stt(audio: Path, base_url: str, api_key: str, language: str) -> dict:
    """POST audio to 60db /stt via curl (dependency-free, like transcribe.sh).
    diarize=true mirrors Soniox enable_speaker_diarization=true.
    """
    proc = subprocess.run([
        "curl", "-s", "-X", "POST", f"{base_url}/stt",
        "-H", f"Authorization: Bearer {api_key}",
        "-F", f"file=@{audio}",
        "-F", f"language={language}",
        "-F", "diarize=true",
        "-F", "return_timestamps=word",
    ], capture_output=True, text=True, check=True)
    if not proc.stdout.strip():
        sys.exit("[60db] empty response from /stt")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.exit(f"[60db] non-JSON response: {proc.stdout[:300]}")
    if data.get("code") or data.get("error"):
        sys.exit(f"[60db] STT error: {data.get('code') or data.get('error')} "
                 f"— {data.get('message', '')}")
    return data


def adapt_tokens(data: dict) -> list:
    """60db response → Soniox-schema token list.

    Iterate segments→words so each word inherits its segment's speaker
    (60db carries speaker at the segment level, not per word). Times go
    seconds→ms. Each token text is space-prefixed so join('') reconstructs a
    naturally spaced transcript, matching Soniox's leading-space convention
    that match_timing.py's substring search relies on.
    """
    segments = data.get("segments") or []

    # Map 60db speaker labels → "1","2",... (sorted, pyannote convention).
    labels = sorted({sp["speaker"]
                     for seg in segments
                     for sp in (seg.get("speakers") or [])})
    label_to_id = {lbl: str(i + 1) for i, lbl in enumerate(labels)}

    def speaker_for(mid_s: float, seg_speakers: list):
        if not seg_speakers:
            return None
        for sp in seg_speakers:
            if sp["start"] <= mid_s < sp["end"]:
                return label_to_id[sp["speaker"]]
        # Word in a gap between speaker turns: attribute to the closest one.
        best = min(seg_speakers,
                   key=lambda sp: min(abs(mid_s - sp["start"]), abs(mid_s - sp["end"])))
        return label_to_id[best["speaker"]]

    tokens = []
    for seg in segments:
        seg_speakers = seg.get("speakers") or []
        for w in seg.get("words") or []:
            start_s, end_s = float(w["start"]), float(w["end"])
            tokens.append({
                "text": " " + w["word"],
                "start_ms": int(round(start_s * 1000)),
                "end_ms": int(round(end_s * 1000)),
                "speaker": speaker_for((start_s + end_s) / 2, seg_speakers),
            })

    # Fallback: no segment-level words (e.g. diarization off) → use flat words,
    # speaker unknown (None), exactly like Soniox with diarization disabled.
    if not tokens and data.get("words"):
        for w in data["words"]:
            start_s, end_s = float(w["start"]), float(w["end"])
            tokens.append({
                "text": " " + w["word"],
                "start_ms": int(round(start_s * 1000)),
                "end_ms": int(round(end_s * 1000)),
                "speaker": None,
            })
    return tokens


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: transcribe_60db.py <video_id>")
    video_id = sys.argv[1]

    repo_root = Path(__file__).resolve().parent.parent
    load_env(repo_root)

    api_key = os.environ.get("SIXTYDB_API_KEY")
    if not api_key:
        sys.exit("SIXTYDB_API_KEY missing — set it in .env")
    base_url = os.environ.get("SIXTYDB_BASE_URL", "https://api.60db.ai").rstrip("/")
    language = os.environ.get("DUBYDUBY_STT_LANG", "en")

    base = repo_root / "output" / video_id
    audio = base / "1_source" / "audio.mp3"
    if not audio.exists():
        sys.exit(f"Audio missing: {audio}")

    out_dir = base / "2_transcript"
    out_dir.mkdir(parents=True, exist_ok=True)

    upload = fit_under_limit(audio)
    try:
        print(f"[60db] POST {base_url}/stt (language={language}, diarize=true)",
              file=sys.stderr)
        data = call_stt(upload, base_url, api_key, language)
    finally:
        if upload != audio:
            upload.unlink(missing_ok=True)

    tokens = adapt_tokens(data)
    if not tokens:
        sys.exit("[60db] no word-level tokens in response — "
                 "check return_timestamps support / audio content")

    (out_dir / "tokens.json").write_text(
        json.dumps({"tokens": tokens}, ensure_ascii=False, indent=2))

    # Human-readable transcript — built from the SAME joined tokens that
    # match_timing.py will search, so the agent translates exactly that text.
    full_text = "".join(t["text"] for t in tokens)
    n_speakers = len({t["speaker"] for t in tokens if t["speaker"]})
    (out_dir / "transcript.md").write_text(
        "# EN transcript (60db)\n\n"
        f"Tokens: {len(tokens)}  ·  Speakers: {n_speakers}  ·  "
        f"Detected language: {data.get('language', language)}\n\n"
        "## Full text\n\n"
        f"{full_text}\n")

    print(f"OK: {out_dir}/{{tokens.json, transcript.md}} "
          f"({len(tokens)} tokens, {n_speakers} speakers)")


if __name__ == "__main__":
    main()
