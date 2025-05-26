import sys
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter

def extract_video_id(url_or_id: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|\/v\/|\/embed\/)([^#&?]{11})", url_or_id)
    return match.group(1) if match else url_or_id  # fallback: assume it's already a video ID

def fetch_human_transcript(video_id: str, output_dir: str):
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)
        transcript = transcript_list.find_manually_created_transcript(['en'])
        data = transcript.fetch()

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{video_id}_official.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json_str = JSONFormatter().format_transcript(data, indent=2)
            f.write(json_str)

        print(f"✅ Human transcript saved to {out_path}")
    except Exception as e:
        print(f"❌ Error fetching transcript: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_official_transcript.py <YouTube_URL_or_ID>")
        sys.exit(1)

    video_input = sys.argv[1]
    video_id = extract_video_id(video_input)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "data", "transcripts")

    fetch_human_transcript(video_id, output_dir)
