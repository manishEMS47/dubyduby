#!/usr/bin/env python3
"""Generate ASS subtitle (YouTube style: white text on rounded semi-transparent box).

Timing strategy:
  Subtitle cues anchor to actual KO audio playback positions (not source EN
  timestamps). Computed from per-utterance wav durations after silenceremove
  + atempo, so subtitles match what the viewer hears.

Reads:
  output/<video_id>/3_translation/utterances.json
  output/<video_id>/4_synth/utt-NNN.wav (durations)
  output/<video_id>/5_intermediate/dub_raw.wav and dub_clean.wav (silence removal)
  output/<video_id>/5_intermediate/atempo.txt

Writes:
  output/<video_id>/6_final/subtitles.ass
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import ImageFont


MAX_CHARS_PER_LINE = 28
MAX_LINES = 2
MAX_BOX_WIDTH_RATIO = 0.85  # cap box width to 85% of video width (force wrap)
FONT_NAME = "Pretendard"
FONT_SIZE = 56
PLAY_RES_X = 1920
PLAY_RES_Y = 1080

PAD_X = 24
PAD_Y = 10
LINE_SPACING = 4
BOX_RADIUS = 8
MARGIN_V = 80

COLOR_TEXT = "&H00FFFFFF"
COLOR_BOX = "&H80000000"

# Per-voice subtitle colors (ASS BGR order: &HAABBGGRR).
# Voice-based (not speaker-id-based) so speakers sharing the same voice
# get the same color — handy when Soniox splits one person into two ids
# (intro narration + dialogue) but they're assigned the same voice.
# Sequence: white → warm yellow → cyan → magenta → mint. White is the
# default for unknown voices.
VOICE_COLOR_PALETTE = [
    "&H0099EEFF",  # yellow  — first distinct voice (host/interviewer)
    "&H00FFFFFF",  # white   — second distinct voice (guest/interviewee, the focus)
    "&H00FFCC99",  # cyan    — third
    "&H00FF99CC",  # magenta — fourth
    "&H0099FFCC",  # mint    — fifth
]
DEFAULT_TEXT_COLOR = "&H00FFFFFF"


def build_voice_color_map(speakers: dict) -> dict:
    """Map speaker_id → color, where speakers sharing a voice share a color."""
    voice_to_color = {}
    palette_idx = 0
    spk_to_color = {}
    # iterate in stable order so the color of "first voice to appear" stays predictable
    for spk in sorted(speakers.keys()):
        voice = speakers[spk].get("voice")
        if voice and voice not in voice_to_color:
            voice_to_color[voice] = VOICE_COLOR_PALETTE[palette_idx % len(VOICE_COLOR_PALETTE)]
            palette_idx += 1
        spk_to_color[spk] = voice_to_color.get(voice, DEFAULT_TEXT_COLOR)
    return spk_to_color

# English secondary subtitle — fixed at top of frame so it never collides
# with the KO subtitle box at the bottom (the two cues run on independent
# timelines: EN follows source-video anchor, KO follows shifted dub timeline,
# so they can be visible simultaneously).
EN_FONT_SIZE = 32                   # ~60% of main
EN_MAX_CHARS_PER_LINE = 60          # English fits more per line
EN_FIXED_TOP_Y = 80                 # px from top of frame (fixed position)
COLOR_EN_TEXT = "&H00DDDDDD"        # slightly dimmer white
COLOR_EN_OUTLINE = "&HA0000000"     # semi-transparent black for readability

FONT_PATH = Path(__file__).parent.parent / "fonts" / "Pretendard-Bold.ttf"
_FONT = ImageFont.truetype(str(FONT_PATH), FONT_SIZE)
_ASCENT, _DESCENT = _FONT.getmetrics()
LINE_HEIGHT = _ASCENT + _DESCENT


def ms_to_ass_time(ms: int) -> str:
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    cs = (ms % 1000) // 10
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _line_width_px(line: str) -> int:
    bbox = _FONT.getbbox(line)
    return bbox[2] - bbox[0]


def wrap_korean(text: str, max_per_line: int = MAX_CHARS_PER_LINE):
    """Wrap by space into <= MAX_LINES, then enforce pixel max width.
    If a line exceeds (PLAY_RES_X * MAX_BOX_WIDTH_RATIO - PAD_X*2), break on space.
    """
    max_px = PLAY_RES_X * MAX_BOX_WIDTH_RATIO - PAD_X * 2
    text = text.strip()
    words = text.split(" ")

    # Greedy pack by pixel width AND char count
    lines = []
    current = []
    for w in words:
        candidate = " ".join(current + [w]) if current else w
        if (
            len(candidate) > max_per_line and current
        ) or _line_width_px(candidate) > max_px:
            if current:
                lines.append(" ".join(current))
                current = [w]
            else:
                # Single very long word — hard split
                lines.append(w[:max_per_line])
                current = [w[max_per_line:]]
        else:
            current.append(w)
    if current:
        lines.append(" ".join(current))

    # Note: NO truncation here anymore. Overflow handling moved to the
    # caller — if total > MAX_LINES, the caller splits the cue into two
    # consecutive time slots so all content stays visible. wrap_korean now
    # just returns however many lines the text needs.
    return lines


def rounded_rect_drawing(w: float, h: float, r: float) -> str:
    c = r * 0.4477
    cmds = [
        f"m {r} 0",
        f"l {w - r} 0",
        f"b {w - c} 0 {w} {c} {w} {r}",
        f"l {w} {h - r}",
        f"b {w} {h - c} {w - c} {h} {w - r} {h}",
        f"l {r} {h}",
        f"b {c} {h} 0 {h - c} 0 {h - r}",
        f"l 0 {r}",
        f"b 0 {c} {c} 0 {r} 0",
    ]
    return " ".join(cmds)


def measure_lines(lines):
    if not lines:
        return PAD_X * 2, PAD_Y * 2
    widths = [(_FONT.getbbox(l)[2] - _FONT.getbbox(l)[0]) for l in lines]
    w = max(widths) + PAD_X * 2
    h = len(lines) * LINE_HEIGHT + (len(lines) - 1) * LINE_SPACING + PAD_Y * 2
    return w, h


def wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as wf:
        return int(wf.getnframes() / wf.getframerate() * 1000)


SILENCE_STOP_DURATION_MS = 200  # must match finalize.sh stop_duration=0.2
# Per-utt leading silence measured directly from wav (Supertonic wavs have
# variable padding 50–500ms). Threshold for "voice" in i16 PCM peak.
VOICE_AMPLITUDE_THRESHOLD = 0.01  # ~ -40dBFS in normalized [-1, 1]


def measure_leading_silence_ms(wav_path: Path) -> int:
    """Find ms from start where amplitude first exceeds threshold."""
    with wave.open(str(wav_path), "rb") as wf:
        n = wf.getnframes()
        sr = wf.getframerate()
        raw = wf.readframes(n)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    threshold = VOICE_AMPLITUDE_THRESHOLD
    above = np.where(np.abs(samples) > threshold)[0]
    if len(above) == 0:
        return 0
    return int(above[0] / sr * 1000)


def _parse_raw_silences(path: Path):
    """Return list of (start_ms, end_ms) silence segments from raw silencedetect."""
    import re
    if not path.exists():
        return []
    starts, ends = [], []
    for line in path.read_text().splitlines():
        m = re.search(r"silence_start:\s*([\d.]+)", line)
        if m:
            starts.append(int(float(m.group(1)) * 1000))
        m2 = re.search(r"silence_end:\s*([\d.]+)", line)
        if m2:
            ends.append(int(float(m2.group(1)) * 1000))
    # Pair them — silencedetect emits start/end in order
    pairs = []
    for s, e in zip(starts, ends):
        if e > s:
            pairs.append((s, e))
    return pairs


def _cumulative_cut(t_raw_ms: int, silences) -> int:
    """How many ms of silence got removed up to position t_raw_ms in raw audio.

    silenceremove behavior: each silence chunk longer than SILENCE_STOP_DURATION
    gets trimmed to SILENCE_STOP_DURATION (≈ 200ms kept).
    """
    cut = 0
    for s_start, s_end in silences:
        if s_end <= t_raw_ms:
            # Whole silence is before t — full cut
            dur = s_end - s_start
            if dur > SILENCE_STOP_DURATION_MS:
                cut += dur - SILENCE_STOP_DURATION_MS
        elif s_start < t_raw_ms < s_end:
            # t is inside this silence — partial. Keep first 200ms, cut rest.
            inside = t_raw_ms - s_start
            cut += max(0, inside - SILENCE_STOP_DURATION_MS)
            break
        else:
            # silence is entirely after t
            break
    return cut


def _parse_final_onsets(path: Path) -> list:
    """Return voice onset times (ms) from silencedetect on dub_final.wav."""
    import re
    if not path.exists():
        return []
    starts = []
    ends = []
    for line in path.read_text().splitlines():
        m = re.search(r"silence_start:\s*([\d.]+)", line)
        if m:
            starts.append(int(float(m.group(1)) * 1000))
        m2 = re.search(r"silence_end:\s*([\d.]+)", line)
        if m2:
            ends.append(int(float(m2.group(1)) * 1000))
    # Voice begins at silence_end (or at 0 if no leading silence)
    onsets = list(ends)
    if not starts or starts[0] > 100:
        onsets.insert(0, 0)
    return onsets


def _snap_to_onset(cue_ms: int, onsets: list, tolerance_ms: int = 500) -> int:
    if not onsets:
        return cue_ms
    nearest = min(onsets, key=lambda o: abs(o - cue_ms))
    return nearest if abs(nearest - cue_ms) <= tolerance_ms else cue_ms


def compute_audio_timing(utterances, base: Path, atempo: float = 1.0) -> list:
    """Use placement.json from place_timeline.py — per-utt placement at
    utterance.start_ms with per-utt atempo. Each cue anchors to source video
    speaker timeline; cue end = placed audio end.
    """
    placement = json.load(open(base / "5_intermediate" / "placement.json"))
    by_idx = {p["i"]: p for p in placement}
    out = []
    for i, u in enumerate(utterances):
        p = by_idx.get(i)
        if not p:
            continue
        # Attach the original English-speaker anchor time so the EN subtitle
        # can stay synced with the source video (mouth movements), while the
        # KO subtitle follows the actual dub playback after adaptive shift.
        u["_anchor_start_ms"] = p.get("anchor_start_ms", p["start_ms"])
        out.append((u, p["start_ms"], p["end_ms"]))

    # Snap each cue start to nearest real voice onset detected on dub_final.wav.
    # Corrects accumulated algorithm drift (silenceremove non-linearity, atempo
    # rounding) by anchoring to ground truth.
    final_onsets = _parse_final_onsets(base / "5_intermediate" / "silence_log_final.txt")
    if final_onsets:
        out = [(u, _snap_to_onset(s, final_onsets), e) for u, s, e in out]

    # Sustain each cue exactly to the start of the next — no gap at all,
    # so subtitles flow continuously without flicker between utterances.
    for i in range(len(out) - 1):
        u, s, e = out[i]
        next_start = out[i + 1][1]
        out[i] = (u, s, max(e, next_start))
    return out


def build_ass(timed_utts, spk_to_color: dict = None) -> str:
    spk_to_color = spk_to_color or {}
    header = f"""[Script Info]
Title: dubyduby
ScriptType: v4.00+
PlayResX: {PLAY_RES_X}
PlayResY: {PLAY_RES_Y}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Text,{FONT_NAME},{FONT_SIZE},{COLOR_TEXT},{COLOR_TEXT},&H00000000&,&H00000000&,1,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1
Style: Box,{FONT_NAME},{FONT_SIZE},{COLOR_BOX},{COLOR_BOX},{COLOR_BOX},{COLOR_BOX},0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: TextEn,{FONT_NAME},{EN_FONT_SIZE},{COLOR_EN_TEXT},{COLOR_EN_TEXT},{COLOR_EN_OUTLINE},&H00000000&,0,1,0,0,100,100,0,0,1,2,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    out = [header]
    # Pre-compute next anchor for English cue end (= next utt's source-video time)
    anchors = [u.get("_anchor_start_ms", s) for u, s, _ in timed_utts]

    def _emit_ko_cue(lines: list, c_start: int, c_end: int, color: str):
        """Emit one KO subtitle cue (rounded box + text) for [c_start, c_end].
        Always exactly 1 or 2 lines."""
        box_w, box_h = measure_lines(lines)
        cx = PLAY_RES_X / 2
        b_bot = PLAY_RES_Y - MARGIN_V
        b_top = b_bot - box_h
        b_left = cx - box_w / 2
        t_cy = b_top + box_h / 2
        drawing = rounded_rect_drawing(box_w, box_h, BOX_RADIUS)
        color_tag = f"\\1c{color}" if color != DEFAULT_TEXT_COLOR else ""
        out.append(
            f"Dialogue: 0,{ms_to_ass_time(c_start)},{ms_to_ass_time(c_end)},Box,,0,0,0,,"
            f"{{\\pos({b_left:.0f},{b_top:.0f})\\an7\\bord0\\shad0\\p1}}{drawing}{{\\p0}}\n"
        )
        text = r"\N".join(lines)
        out.append(
            f"Dialogue: 1,{ms_to_ass_time(c_start)},{ms_to_ass_time(c_end)},Text,,0,0,0,,"
            f"{{\\pos({cx:.0f},{t_cy:.0f})\\an5{color_tag}}}{text}\n"
        )

    for idx, (u, start_ms, end_ms) in enumerate(timed_utts):
        if end_ms <= start_ms:
            end_ms = start_ms + 1000
        lines = wrap_korean(u["text"])
        spk = u.get("speaker")
        color = spk_to_color.get(spk, DEFAULT_TEXT_COLOR)

        # If KO needs >2 lines, split the time slot in half and show first
        # half of lines in cue A, second half in cue B. Audio stays one
        # continuous utterance; only the subtitle visually splits.
        if len(lines) > MAX_LINES:
            split_count = (len(lines) + MAX_LINES - 1) // MAX_LINES  # number of cue parts
            chunk_size = (len(lines) + split_count - 1) // split_count
            slot_ms = end_ms - start_ms
            for part in range(split_count):
                seg_start = start_ms + part * slot_ms // split_count
                seg_end = start_ms + (part + 1) * slot_ms // split_count
                if part == split_count - 1:
                    seg_end = end_ms
                seg_lines = lines[part * chunk_size : (part + 1) * chunk_size]
                if not seg_lines:
                    continue
                _emit_ko_cue(seg_lines, seg_start, seg_end, color)
        else:
            _emit_ko_cue(lines, start_ms, end_ms, color)

        # English secondary subtitle at FIXED top of frame. Uses ORIGINAL
        # source-video timing (anchor) so it stays synced with the speaker's
        # mouth in the picture, independent of the KO cue at the bottom.
        en_text = u.get("text_en", "").strip()
        if en_text:
            en_start_ms = u.get("_anchor_start_ms", start_ms)
            en_end_ms = anchors[idx + 1] if idx + 1 < len(anchors) else en_start_ms + 3000
            if en_end_ms <= en_start_ms:
                en_end_ms = en_start_ms + 1000
            en_lines = wrap_english(en_text)
            en_text_block = r"\N".join(en_lines)
            en_color_tag = f"\\1c{color}" if color != DEFAULT_TEXT_COLOR else ""
            en_cx = PLAY_RES_X / 2
            # an8 = top-center alignment, so y = absolute distance from top
            out.append(
                f"Dialogue: 2,{ms_to_ass_time(en_start_ms)},{ms_to_ass_time(en_end_ms)},TextEn,,0,0,0,,"
                f"{{\\pos({en_cx:.0f},{EN_FIXED_TOP_Y})\\an8{en_color_tag}}}{en_text_block}\n"
            )
    return "".join(out)


def wrap_english(text: str, max_per_line: int = EN_MAX_CHARS_PER_LINE):
    """Greedy wrap by word, capped at 2 lines. Truncate the rest with ellipsis."""
    text = text.strip()
    if len(text) <= max_per_line:
        return [text]
    words = text.split(" ")
    lines = []
    current = []
    for w in words:
        candidate = " ".join(current + [w]) if current else w
        if len(candidate) > max_per_line and current:
            lines.append(" ".join(current))
            current = [w]
            if len(lines) >= MAX_LINES - 1:
                # last line — fit what we can, ellipsis if overflowing
                rest = words[words.index(w):]
                last = " ".join(rest)
                if len(last) > max_per_line:
                    last = last[: max_per_line - 1].rsplit(" ", 1)[0] + "…"
                lines.append(last)
                return lines
        else:
            current.append(w)
    if current:
        lines.append(" ".join(current))
    return lines


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: subtitle.py <video_id>")
    video_id = sys.argv[1]
    base = Path(f"output/{video_id}")
    utts = json.load(open(base / "3_translation" / "utterances.json"))

    # Attach English source text for the bilingual subtitle.
    # match_timing.py now stores text_en + sent_idx directly on each utterance,
    # so this fallback only fires for older utterances.json files where those
    # fields are missing.
    if utts and "text_en" not in utts[0]:
        sents_path = base / "3_translation" / "sentences.json"
        if sents_path.exists():
            sents = json.load(open(sents_path))
            # Use sent_idx if present, else fall back to positional 1:1
            for i, u in enumerate(utts):
                idx = u.get("sent_idx", i)
                if 0 <= idx < len(sents):
                    u["text_en"] = sents[idx].get("en", "")

    # speaker → color (voice-based, so same person across speaker_ids = same color)
    spk_to_color = {}
    speakers_path = base / "2_transcript" / "speakers.json"
    if speakers_path.exists():
        speakers = json.load(open(speakers_path))
        spk_to_color = build_voice_color_map(speakers)

    timed = compute_audio_timing(utts, base)
    out = base / "6_final" / "subtitles.ass"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_ass(timed, spk_to_color))
    print(f"[subtitle] {len(timed)} cues → {out}")
    if spk_to_color:
        unique_colors = set(spk_to_color.values())
        print(f"[subtitle] {len(unique_colors)} distinct speaker colors applied")


if __name__ == "__main__":
    main()
