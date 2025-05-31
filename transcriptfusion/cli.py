#!/usr/bin/env python3
"""
TranscriptFusion CLI - Command line interface for YouTube transcript processing
"""

import click
import os
import sys
from pathlib import Path

@click.group()
def cli():
    """TranscriptFusion - YouTube transcript processing pipeline"""
    pass

@cli.command()
@click.argument('url')
@click.option('--output-dir', default='data/audio_clips', help='Output directory for audio files')
def download(url, output_dir):
    """Download audio from YouTube URL"""
    click.echo(f"Downloading audio from: {url}")
    # Import here to avoid startup errors
    from transcriptfusion.core.download_youtube_audio import download_audio
    download_audio(url, output_dir)
    click.echo("✅ Audio downloaded successfully!")

@cli.command()
@click.argument('video_id')
@click.option('--output-dir', default='data/transcripts', help='Output directory for transcripts')
def fetch(video_id, output_dir):
    """Fetch official YouTube transcript"""
    click.echo(f"Fetching transcript for: {video_id}")
    # Import here to avoid startup errors
    from transcriptfusion.core.fetch_youtube_transcript import fetch_transcript
    fetch_transcript(video_id, output_dir)
    click.echo("✅ Transcript fetched successfully!")

@cli.command()
@click.argument('audio_path')
@click.option('--output-dir', default='data/transcripts', help='Output directory for transcripts')
def transcribe(audio_path, output_dir):
    """Transcribe audio using WhisperX"""
    click.echo(f"Transcribing: {audio_path}")
    # Import here to avoid startup errors
    from transcriptfusion.core.transcribe_with_whisperx import transcribe_audio
    transcribe_audio(audio_path, output_dir)
    click.echo("✅ Audio transcribed successfully!")

@cli.command()
@click.argument('video_id')
@click.option('--dev', is_flag=True, help='Use development mode')
@click.option('--no-qwen', is_flag=True, help='Disable Qwen3 for alignment')
def enrich(video_id, dev, no_qwen):
    """Enrich transcript with both sources"""
    click.echo(f"Enriching transcript for: {video_id}")
    # Import and use the ArticuLoopTimingEnricher class directly
    from transcriptfusion.core.enrich_transcript import ArticuLoopTimingEnricher
    
    enricher = ArticuLoopTimingEnricher(dev_mode=dev, use_qwen=not no_qwen)
    success = enricher.process_video(video_id)
    
    if success:
        click.echo("✅ Transcript enriched successfully!")
    else:
        click.echo("❌ Enrichment failed!", err=True)
        sys.exit(1)

@cli.command(name='full-pipeline')
@click.argument('url')
@click.option('--dev', is_flag=True, help='Use development mode')
def full_pipeline(url, dev):
    """Run the complete pipeline on a YouTube URL"""
    # Extract video ID from URL
    if 'youtube.com/watch?v=' in url:
        video_id = url.split('v=')[1].split('&')[0]
    elif 'youtu.be/' in url:
        video_id = url.split('/')[-1].split('?')[0]
    else:
        click.echo("❌ Invalid YouTube URL", err=True)
        return
    
    click.echo(f"🎬 Running full pipeline for video: {video_id}")
    
    try:
        # Import functions as needed
        from transcriptfusion.core.download_youtube_audio import download_audio
        from transcriptfusion.core.fetch_youtube_transcript import fetch_transcript
        from transcriptfusion.core.transcribe_with_whisperx import transcribe_audio
        from transcriptfusion.core.enrich_transcript import ArticuLoopTimingEnricher
        
        # Step 1: Download audio
        click.echo("\n1/4 Downloading audio...")
        download_audio(url)
        
        # Step 2: Fetch official transcript
        click.echo("\n2/4 Fetching official transcript...")
        project_root = os.getcwd()
        output_dir = os.path.join(project_root, "data", "transcripts")
        fetch_transcript(video_id, output_dir)
        
        # Step 3: Transcribe with WhisperX
        click.echo("\n3/4 Transcribing with WhisperX...")
        audio_path = os.path.join(project_root, "data", "audio_clips", f"{video_id}.mp3")
        transcribe_audio(audio_path)
        
        # Step 4: Enrich transcript
        click.echo("\n4/4 Enriching transcript...")
        enricher = ArticuLoopTimingEnricher(dev_mode=dev)
        success = enricher.process_video(video_id)
        
        if success:
            click.echo(f"\n✅ Pipeline complete! Check data/transcripts/{video_id}_enriched.json")
        else:
            raise Exception("Enrichment failed")
        
    except Exception as e:
        click.echo(f"❌ Pipeline failed: {str(e)}", err=True)
        sys.exit(1)

# Entry point
def main():
    cli()

if __name__ == '__main__':
    main()