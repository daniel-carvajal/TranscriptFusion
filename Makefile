.PHONY: help install install-dev test run clean clean-all uninstall

# Configuration
PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
DEFAULT_VIDEO_ID := epVW0_iVBX8

# Color codes for better output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

help:
	@echo "$(GREEN)TranscriptFusion Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  make install        # Setup everything"
	@echo "  make test          # Run test video"
	@echo ""
	@echo "$(YELLOW)Installation:$(NC)"
	@echo "  install            # Complete setup (recommended)"
	@echo "  install-dev        # Install as global command only"
	@echo "  uninstall          # Remove global installation"
	@echo ""
	@echo "$(YELLOW)Usage:$(NC)"
	@echo "  test               # Process test video ($(DEFAULT_VIDEO_ID))"
	@echo "  run VIDEO_ID=xxx   # Process specific video"
	@echo ""
	@echo "$(YELLOW)Maintenance:$(NC)"
	@echo "  clean              # Remove generated files"
	@echo "  clean-all          # Remove everything (including venv)"

# Complete installation (recommended)
install: _check-python
	@echo "$(YELLOW)Setting up TranscriptFusion...$(NC)"
	@$(MAKE) -s _setup-venv
	@$(MAKE) -s _install-deps
	@$(MAKE) -s install-dev
	@echo ""
	@echo "$(GREEN)✅ Installation complete!$(NC)"
	@echo "Try: $(YELLOW)make test$(NC) or $(YELLOW)transcriptfusion --help$(NC)"

# Install as global command (development mode)
# Note: If using pyenv-virtualenv, this installs to your active environment
install-dev: _check-setup-py
	@echo "Installing TranscriptFusion globally..."
	@pip install -e . --quiet
	@echo "$(GREEN)✅ Global command 'transcriptfusion' installed$(NC)"

# Quick test with default video
test:
	@if command -v transcriptfusion >/dev/null 2>&1; then \
		echo "$(YELLOW)Running test with video: $(DEFAULT_VIDEO_ID)$(NC)"; \
		transcriptfusion full-pipeline https://youtube.com/watch?v=$(DEFAULT_VIDEO_ID) --dev; \
	else \
		echo "$(RED)❌ TranscriptFusion not installed. Run 'make install' first$(NC)"; \
		exit 1; \
	fi

# Run with specific video ID
run:
ifndef VIDEO_ID
	@echo "$(RED)❌ VIDEO_ID required$(NC)"
	@echo "Usage: make run VIDEO_ID=your_video_id"
	@exit 1
endif
	@if command -v transcriptfusion >/dev/null 2>&1; then \
		echo "$(YELLOW)Processing video: $(VIDEO_ID)$(NC)"; \
		transcriptfusion full-pipeline https://youtube.com/watch?v=$(VIDEO_ID) --dev; \
	else \
		echo "$(RED)❌ TranscriptFusion not installed. Run 'make install' first$(NC)"; \
		exit 1; \
	fi

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -rf data/audio_clips/*.mp3 data/transcripts/*.json logs/*.txt 2>/dev/null || true
	@echo "$(GREEN)✅ Cleaned$(NC)"

# Clean everything including venv
clean-all: clean uninstall
	@echo "Removing virtual environment..."
	@rm -rf $(VENV) 2>/dev/null || true
	@echo "$(GREEN)✅ Everything cleaned$(NC)"

# Uninstall global installation
uninstall:
	@if pip show transcriptfusion >/dev/null 2>&1; then \
		echo "Uninstalling TranscriptFusion..."; \
		pip uninstall transcriptfusion -y >/dev/null 2>&1; \
		echo "$(GREEN)✅ Uninstalled$(NC)"; \
	else \
		echo "TranscriptFusion not installed globally"; \
	fi

# --- Private targets (prefixed with _) ---

# Check if Python is available
_check-python:
	@command -v $(PYTHON) >/dev/null 2>&1 || { \
		echo "$(RED)❌ Python3 not found. Please install Python 3.8+$(NC)"; \
		exit 1; \
	}

# Check if setup.py exists
_check-setup-py:
	@if [ ! -f setup.py ]; then \
		echo "$(RED)❌ setup.py not found in current directory$(NC)"; \
		exit 1; \
	fi

# Create virtual environment
_setup-venv:
	@if [ ! -d $(VENV) ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV); \
	fi

# Install dependencies
_install-deps: $(VENV)/bin/activate
	@echo "Installing dependencies..."
	@$(PIP) install --upgrade pip --quiet
	@$(PIP) install -r requirements.txt --quiet

# Ensure venv exists
$(VENV)/bin/activate:
	@$(MAKE) -s _setup-venv