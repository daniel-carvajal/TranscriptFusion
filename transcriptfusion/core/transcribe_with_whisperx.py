import subprocess
import sys
import os

def transcribe_audio(audio_file: str, output_dir: str = None):
    # Resolve output path relative to the script's parent directory (project root)
    project_root = os.getcwd()
    if output_dir is None:
        output_dir = os.path.join(project_root, "data", "transcripts")

    os.makedirs(output_dir, exist_ok=True)

    command = [
        "whisperx",
        audio_file,
        "--output_format", "json",
        "--output_dir", output_dir,
        "--compute_type", "float32",
        "--device", "cpu",
        "--model", "large", # tiny, base, small, medium, large, turbo
        "--language", "en"
    ]

    try:
        print(f"Transcribing {audio_file} with WhisperX...")
        subprocess.run(command, check=True)
        print(f"Transcription complete. Output saved in: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe_with_whisperx.py <path_to_audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    transcribe_audio(audio_path)
