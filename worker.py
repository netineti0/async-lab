import json
import sys
from pathlib import Path
import subprocess
import assemblyai as aai
from assemblyai import TranscriptionConfig
import os
import traceback
import time
import hashlib

SHARD = int(sys.argv[1])
TOTAL = int(sys.argv[2])
channel_name = sys.argv[3]

TRANSCRIPTS = Path("transcripts")

api_key = os.environ.get(f"ASSEMBLYAI_API_KEY")
if not api_key:
    raise ValueError("ASSEMBLYAI_API_KEY environment variable not set")
aai.settings.api_key = api_key

config = aai.TranscriptionConfig(
  speech_models=[
  "universal-3-5-pro"
],
  speaker_labels=False,
  format_text=True,
  punctuate=True,
  language_code="hi"
)

def commit_video(channel: str, video_id: str):
    """Commit and push a single processed video with concurrency handling."""
    transcript_path = TRANSCRIPTS / channel / f"{video_id}.json"

    # 1. Stage the specific file we want to commit
    subprocess.run(["git", "add", str(transcript_path)], check=True)

    # 2. Commit it locally first.
    subprocess.run(["git", "commit", "-m", f"transcript: {channel}/{video_id} [shard: {SHARD}]"], check=True)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Stash any other unstaged files so workspace is clean
            subprocess.run(["git", "stash", "--include-untracked"], check=True)

            # Pull and rebase. If an add/add conflict happens, `-Xtheirs` tells 
            # Git to accept the remote file already pushed to origin/main.
            subprocess.run(
                ["git", "pull", "--rebase", "-Xtheirs", "origin", "main"],
                check=True
            )

            # Pop the stash back
            subprocess.run(["git", "stash", "pop"], check=False)

            # Push our commit
            subprocess.run(["git", "push", "origin", "main"], check=True)
            print(f"Successfully pushed transcript for {video_id}")
            return

        except subprocess.CalledProcessError:
            print(f"Conflict or push collision detected. Cleaning up and retrying ({attempt + 1}/{max_retries})...")
            # CRITICAL: Always abort any stuck rebase before looping to try again
            subprocess.run(["git", "rebase", "--abort"], check=False)
            time.sleep(2)

    raise RuntimeError(f"Failed to push transcript for {video_id} after {max_retries} attempts.")


def belongs(video_id: str) -> bool:
    # Use MD5 to get a stable, consistent integer hash across all machines
    hasher = hashlib.md5(video_id.encode('utf-8'))
    hash_int = int(hasher.hexdigest(), 16)
    return hash_int % TOTAL == SHARD


def is_processed(channel: str, video_id: str) -> bool:
    """Check if video already has a transcript."""
    return (TRANSCRIPTS / channel / f"{video_id}.json").exists()


def download_audio(video_id: str, channel: str) -> Path:
    """Download best available audio."""
    audio_dir = Path("audio") / channel
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Don't preset extension - let yt-dlp decide
    url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"downloading audio {video_id}")

    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "best",
        "--audio-quality", "0",
        "-o", str(audio_dir / video_id),  # No extension
        "--cookies", f"cookies.txt",
        "--no-playlist",
        url
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"downloaded audio {video_id}")

    # Find what file was created
    for f in audio_dir.glob(f"{video_id}.*"):
        return f

    raise Exception(f"No file created for {video_id}")


def transcribe(audio_filepath):
    transcriber = aai.Transcriber(config=config)

    max_retries = 3
    base_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            print(f"transcribing audio {audio_filepath} (Attempt {attempt + 1}/{max_retries})")
            transcript = transcriber.transcribe(str(audio_filepath))
            print(f"transcribed audio {audio_filepath}")
            return transcript.json_response

        except Exception as e:
            error_msg = str(e).lower()
            # Catch 429 rate limits, concurrency blocks, or brief server timeouts
            if "limit exceeded" in error_msg or "429" in error_msg or "timeout" in error_msg:
                if attempt < max_retries - 1:
                    # Exponential backoff: 5s, then 10s, etc.
                    delay = base_delay * (attempt + 1)
                    print(f"AssemblyAI rate/concurrency limit hit. Retrying in {delay}s... Error: {e}")
                    time.sleep(delay)
                    continue

            # If it's a completely different error (e.g., corrupted audio format), raise it immediately
            print(f"Permanent error during transcription: {e}")
            raise e

    raise RuntimeError(f"Failed to transcribe {audio_filepath} after {max_retries} attempts due to API limits.")


def process(channel, vid):
    out = TRANSCRIPTS / channel / f"{vid}.json"

    # This check is now redundant with is_processed(), 
    # but keeping it as a safety check
    if out.exists():
        return

    audio = download_audio(vid, channel)
    transcript = transcribe(audio)

    out.parent.mkdir(parents=True, exist_ok=True)

    # Write atomically to avoid partial files
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(transcript))
    tmp.replace(out)

    audio.unlink(missing_ok=True)


def get_pending_videos(channel: str):
    """Yield video_ids for a specific channel."""
    pending_file = Path(f"{channel}_pending_video_ids.json")

    if not pending_file.exists():
        return

    vids = json.loads(pending_file.read_text())
    for vid in vids:
        yield channel, vid


def main():
    processed_count = 0
    for channel, vid in get_pending_videos(channel_name):
        # Skip if not for this shard
        if not belongs(vid):
            print(f"skipping {vid} from {channel} not belonging to shard {SHARD}")
            continue

        # Skip if already processed (transcript exists)
        if is_processed(channel, vid):
            print(f"skipping already processed {vid} from {channel} in {SHARD}")
            continue

        try:
            print(f"processing {vid} from {channel} in {SHARD}")
            process(channel, vid)
            processed_count += 1

            # Commit immediately after each successful video
            commit_video(channel, vid)

            print(f"✓ Processed {channel}/{vid} (total: {processed_count})")

        except aai.AssemblyAIError as e:
            print(e.status_code)
            raise e

        except Exception as e:
            print(f"✗ Failed to process {channel}/{vid}: {e}")
            traceback.print_exc()
            # Continue with next video
            continue

    print(f"Shard {SHARD} complete. Processed {processed_count} videos.")


if __name__ == "__main__":
    main()
