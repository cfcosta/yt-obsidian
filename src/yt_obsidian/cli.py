import argparse
import json
import re
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit
from typing import Dict, List, Any

import yaml

import yt_dlp
from dotenv import load_dotenv
from litellm import completion
from pydub import AudioSegment
from faster_whisper import WhisperModel

load_dotenv()


def download_audio(url: str, output_dir: Path) -> tuple[Path, Dict[str, Any]]:
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_path = Path(ydl.prepare_filename(info)).with_suffix(".wav")

    return audio_path, info


def load_whisper_model():
    model_size = "base"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return model


def transcribe_with_diarization(audio_path: Path, model) -> List[Dict[str, Any]]:
    segments, info = model.transcribe(str(audio_path), beam_size=5, language="en")

    transcript = []
    for segment in segments:
        transcript.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            }
        )

    transcript_with_speakers = diarize_speakers(audio_path, transcript)

    return transcript_with_speakers


def diarize_speakers(
    audio_path: Path, transcript: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    audio = AudioSegment.from_wav(str(audio_path))

    speakers = ["Speaker A", "Speaker B", "Speaker C", "Speaker D"]
    current_speaker = 0
    last_change_time = 0
    min_duration = 30

    for segment in transcript:
        if segment["start"] - last_change_time > min_duration:
            current_speaker = (current_speaker + 1) % len(speakers)
            last_change_time = segment["start"]
        segment["speaker"] = speakers[current_speaker]

    return transcript


def format_upload_date(date_str: str | None) -> str:
    if not date_str:
        return ""
    if isinstance(date_str, (int, float)):
        date_str = str(date_str)
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return str(date_str)


def select_best_thumbnail(video_info: Dict[str, Any]) -> str:
    thumbnails = video_info.get("thumbnails") or []
    if thumbnails:
        best = max(
            thumbnails,
            key=lambda t: (t.get("width") or 0) * (t.get("height") or 0),
        )
        return best.get("url") or ""
    return video_info.get("thumbnail", "")


def extract_video_metadata(video_info: Dict[str, Any]) -> Dict[str, Any]:
    duration_seconds = video_info.get("duration") or 0
    return {
        "original_url": video_info.get("webpage_url", ""),
        "upload_date": format_upload_date(video_info.get("upload_date")),
        "uploader": video_info.get("uploader", ""),
        "uploader_id": video_info.get("uploader_id", ""),
        "channel": video_info.get("channel") or video_info.get("uploader", ""),
        "channel_url": video_info.get("channel_url") or video_info.get("uploader_url", ""),
        "duration_seconds": duration_seconds,
        "duration_minutes": round(duration_seconds / 60, 2) if duration_seconds else 0,
        "view_count": video_info.get("view_count", 0),
        "like_count": video_info.get("like_count", 0),
        "categories": video_info.get("categories") or [],
        "video_tags": video_info.get("tags") or [],
        "description": (video_info.get("description") or "").strip(),
        "thumbnail_url": select_best_thumbnail(video_info),
    }


def download_thumbnail(thumbnail_url: str, output_path: Path) -> Path | None:
    if not thumbnail_url:
        return None

    try:
        suffix = Path(urlsplit(thumbnail_url).path).suffix or ".jpg"
    except Exception:
        suffix = ".jpg"

    thumbnail_path = output_path.with_name(f"{output_path.stem}-thumb{suffix}")

    try:
        with urllib.request.urlopen(thumbnail_url) as response:
            thumbnail_path.write_bytes(response.read())
        return thumbnail_path
    except Exception:
        return None


def extract_metadata(
    video_info: Dict[str, Any], transcript: List[Dict[str, Any]]
) -> Dict[str, Any]:
    transcript_text = "\n".join([f"{s['speaker']}: {s['text']}" for s in transcript])
    transcript_preview = transcript_text[:2000]

    prompt = f"""Analyze this YouTube video transcript and extract metadata in JSON format.

Video Title: {video_info.get("title", "N/A")}
Video Description: {video_info.get("description", "")[:1000]}

Transcript Preview:
{transcript_preview}

Extract and return ONLY a JSON object with these exact keys:
- title: Main topic/title summary
- summary: Brief 2-3 sentence summary
- tags: List of 5-10 relevant tags
- category: Content category (e.g., Tutorial, Interview, Lecture)
- key_topics: List of main topics discussed
- speakers: List of unique speakers (identify by names if mentioned, otherwise use generic labels)
- duration: Duration in minutes
- language: Content language
- difficulty_level: Beginner/Intermediate/Advanced (if applicable)

Return JSON only, no other text."""

    response = completion(
        model="openrouter/minimax/minimax-m2.1",
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        metadata = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        metadata = {
            "title": video_info.get("title", "Untitled"),
            "summary": "Summary not available",
            "tags": ["video", "youtube"],
            "category": "General",
            "key_topics": [],
            "speakers": ["Speaker A"],
            "duration": round(video_info.get("duration", 0) / 60, 2),
            "language": "en",
            "difficulty_level": "Intermediate",
        }

    metadata.update(
        {
            "original_url": video_info.get("webpage_url", ""),
            "upload_date": video_info.get("upload_date", ""),
            "uploader": video_info.get("uploader", ""),
            "view_count": video_info.get("view_count", 0),
            "like_count": video_info.get("like_count", 0),
        }
    )

    return metadata


def sanitize_filename_component(text: str) -> str:
    """Create a bash-friendly slug from a filename component."""
    text = text.replace(" ", "-")
    text = re.sub(r"[^A-Za-z0-9._-]", "", text)
    text = re.sub(r"-{2,}", "-", text)
    text = text.strip("-_.")
    return text or "untitled"


def build_base_filename(
    video_info: Dict[str, Any], metadata: Dict[str, Any] | None = None
) -> str:
    """Combine channel and title into a sanitized base filename."""
    channel = video_info.get("channel") or video_info.get("uploader") or "channel"
    title = (
        (metadata or {}).get("title")
        or video_info.get("title")
        or video_info.get("fulltitle")
        or "untitled"
    )

    channel_slug = sanitize_filename_component(channel)
    title_slug = sanitize_filename_component(title)

    parts = [p for p in (channel_slug, title_slug) if p]
    return "-".join(parts) if parts else "untitled"


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def update_speaker_names(
    transcript: List[Dict[str, Any]], metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:
    speakers = metadata.get("speakers", [])
    if not speakers:
        return transcript

    generic_labels = ["Speaker A", "Speaker B", "Speaker C", "Speaker D"]
    label_mapping = {}

    for i, speaker_name in enumerate(speakers):
        if i < len(generic_labels):
            label_mapping[generic_labels[i]] = speaker_name

    for segment in transcript:
        if segment["speaker"] in label_mapping:
            segment["speaker"] = label_mapping[segment["speaker"]]

    return transcript


def generate_markdown(
    metadata: Dict[str, Any], transcript: List[Dict[str, Any]], output_path: Path
) -> None:
    def _yamlable(value: Any):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: _yamlable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_yamlable(v) for v in value]
        return value

    yaml_frontmatter = yaml.safe_dump(
        _yamlable(metadata), sort_keys=False, allow_unicode=False
    ).strip()

    content = ["---", yaml_frontmatter, "---", "", f"# {metadata.get('title', 'Untitled')}", ""]

    if metadata.get("thumbnail_path") or metadata.get("thumbnail_url"):
        content.append("## Thumbnail")
        thumb_path = metadata.get("thumbnail_path")
        thumb_url = metadata.get("thumbnail_url")
        if thumb_path:
            content.append(f"![]({thumb_path})")
        elif thumb_url:
            content.append(f"![]({thumb_url})")
        content.append("")

    content.append("## Summary")
    content.append(metadata.get("summary", ""))
    content.append("")

    content.append("## Video Metadata")
    video_lines = [
        f"- Original URL: {metadata.get('original_url', '')}",
        f"- Upload date: {metadata.get('upload_date', '')}",
        f"- Channel: {metadata.get('channel', '')} ({metadata.get('channel_url', '')})",
        f"- Uploader: {metadata.get('uploader', '')} ({metadata.get('uploader_id', '')})",
        f"- Duration: {metadata.get('duration_minutes', 0)} minutes ({metadata.get('duration_seconds', 0)} seconds)",
        f"- Views: {metadata.get('view_count', 0)} | Likes: {metadata.get('like_count', 0)}",
    ]
    if metadata.get("categories"):
        video_lines.append(f"- Categories: {', '.join(metadata.get('categories', []))}")
    if metadata.get("video_tags"):
        video_lines.append(f"- Video tags: {', '.join(metadata.get('video_tags', []))}")
    if metadata.get("description"):
        video_lines.append("- Description:")
        video_lines.append(metadata.get("description", ""))

    content.extend(video_lines)
    content.append("")

    content.append("## Transcript")
    content.append("")
    content.append("| Time | Speaker | Text |")
    content.append("|------|---------|------|")
    for segment in transcript:
        time_str = format_timestamp(segment["start"])
        content.append(f"| {time_str} | {segment['speaker']} | {segment['text']} |")

    content.append("")

    output_path.write_text("\n".join(content))


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe YouTube video and generate Obsidian markdown"
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("-o", "--output", help="Output markdown file path", type=Path)
    parser.add_argument(
        "--keep-audio", action="store_true", help="Keep downloaded audio file"
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        print("Downloading audio...")
        audio_path, video_info = download_audio(args.url, tmpdir_path)
        audio_base_name = build_base_filename(video_info)
        sanitized_audio_path = tmpdir_path / f"{audio_base_name}{audio_path.suffix}"
        if sanitized_audio_path != audio_path:
            audio_path.rename(sanitized_audio_path)
            audio_path = sanitized_audio_path
        print(f"Audio downloaded to: {audio_path}")

        print("Loading Whisper model...")
        model = load_whisper_model()

        print("Transcribing with diarization...")
        transcript = transcribe_with_diarization(audio_path, model)
        print(f"Transcription complete: {len(transcript)} segments")

        print("Extracting metadata with LLM...")
        metadata = extract_metadata(video_info, transcript)
        video_metadata = extract_video_metadata(video_info)
        metadata.update(video_metadata)

        print("Updating speaker names in transcript...")
        transcript = update_speaker_names(transcript, metadata)

        default_output = Path(f"{build_base_filename(video_info, metadata)}.md")
        output_path = args.output or default_output
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print("Downloading thumbnail...")
        thumbnail_path = download_thumbnail(metadata.get("thumbnail_url", ""), output_path)
        if thumbnail_path:
            metadata["thumbnail_path"] = thumbnail_path.name

        print("Generating markdown...")
        generate_markdown(metadata, transcript, output_path)

        if not args.keep_audio:
            audio_path.unlink()

    print(f"Markdown file created: {output_path}")


if __name__ == "__main__":
    main()
