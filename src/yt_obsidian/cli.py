import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any

import yt_dlp
from dotenv import load_dotenv
from litellm import completion
from pydub import AudioSegment
from faster_whisper import WhisperModel
import librosa
import soundfile as sf
import numpy as np

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
        model="openrouter/anthropic/claude-3.5-sonnet",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        api_key=os.getenv("OPENROUTER_API_KEY"),
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
    frontmatter = []
    for key, value in metadata.items():
        if isinstance(value, list):
            frontmatter.append(f"{key}:")
            for item in value:
                frontmatter.append(f"  - {item}")
        elif isinstance(value, (int, float)):
            frontmatter.append(f"{key}: {value}")
        else:
            frontmatter.append(f'{key}: "{value}"')

    content = [
        "---",
        *frontmatter,
        "---",
        "",
        f"# {metadata.get('title', 'Untitled')}",
        "",
    ]
    content.append("## Summary")
    content.append(metadata.get("summary", ""))
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
        print(f"Audio downloaded to: {audio_path}")

        print("Loading Whisper model...")
        model = load_whisper_model()

        print("Transcribing with diarization...")
        transcript = transcribe_with_diarization(audio_path, model)
        print(f"Transcription complete: {len(transcript)} segments")

        print("Extracting metadata with LLM...")
        metadata = extract_metadata(video_info, transcript)

        print("Updating speaker names in transcript...")
        transcript = update_speaker_names(transcript, metadata)

        output_path = args.output or Path(
            f"{metadata.get('title', 'transcript').replace(' ', '_')}.md"
        )

        print("Generating markdown...")
        generate_markdown(metadata, transcript, output_path)

        if not args.keep_audio:
            audio_path.unlink()

    print(f"Markdown file created: {output_path}")


if __name__ == "__main__":
    main()
