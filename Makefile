.PHONY: setup install test-run run clean help

# Default Python version
PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

# Default test video ID (only used in test-run)
DEFAULT_VIDEO_ID := epVW0_iVBX8

help:
	@echo "TranscriptFusion Commands:"
	@echo "  setup     - Create virtual environment"
	@echo "  activate  - Activate virtual environment"
	@echo "  install   - Install dependencies"
	@echo "  run       - Run pipeline with VIDEO_ID=xxx (required)"
	@echo "  test-run  - Run pipeline with default test video ID"
	@echo "  clean     - Remove generated files"
	@echo ""
	@echo "Quick start: make setup && make install && make test-run"

setup-env:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "✅ Virtual environment created in $(VENV)/"

activate-env:
	@echo "To activate venv, run:"
	@echo "source $(VENV)/bin/activate"

install-deps: $(VENV)/bin/activate
	@echo "Installing requirements..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

run: $(VENV)/bin/activate
ifndef VIDEO_ID
	$(error ❌ VIDEO_ID is required. Usage: make run VIDEO_ID=your_id)
endif
	@echo "🎬 Running full pipeline with video ID: $(VIDEO_ID)"
	@echo "1/4 Downloading audio..."
	$(PYTHON_VENV) data-processing/download_youtube_audio.py https://youtube.com/watch?v=$(VIDEO_ID)
	@echo "2/4 Fetching official transcript..."
	$(PYTHON_VENV) data-processing/fetch_official_transcript.py $(VIDEO_ID)
	@echo "3/4 Transcribing with WhisperX..."
	$(PYTHON_VENV) data-processing/transcribe_with_whisperx.py data/audio_clips/$(VIDEO_ID).mp3
	@echo "4/4 Enriching transcript..."
	$(PYTHON_VENV) data-processing/enrich_transcript.py $(VIDEO_ID) --dev
	@echo "✅ Pipeline complete! Check data/transcripts/$(VIDEO_ID)_enriched.json"

test-run:
	@$(MAKE) run VIDEO_ID=$(DEFAULT_VIDEO_ID)

clean:
	@echo "Cleaning up generated files..."
	rm -rf data/audio_clips/*.mp3
	rm -rf data/transcripts/*.json
	rm -rf logs/*.txt
	@echo "✅ Cleaned data files"

# Catch when venv isn't set up
$(VENV)/bin/activate:
	@echo "❌ Virtual environment not found. Run 'make setup' first."
	@exit 1
