import subprocess
import sys
import os

def download_audio(youtube_url: str, output_dir: str = None):
    # Resolve path relative to the script's parent directory (project root)
    project_root = os.getcwd()
    if output_dir is None:
        output_dir = os.path.join(project_root, "data", "audio_clips")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "%(id)s.%(ext)s")

    command = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "mp3",
        "--audio-quality", "0",  # Best quality
        "-o", output_path,
        youtube_url
    ]

    try:
        print(f"Downloading audio from {youtube_url}...")
        subprocess.run(command, check=True)
        print("Download complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_youtube_audio.py <youtube_url>")
        sys.exit(1)

    url = sys.argv[1]
    download_audio(url)
