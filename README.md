# TranscriptFusion

**Precision word-level timing enrichment for human-curated transcripts**

TranscriptFusion combines the accuracy of human-verified YouTube transcripts with the precise word-level timing data from WhisperX to create enriched transcripts with both perfect text and microsecond-accurate word boundaries.

## 🎯 What it does

- Takes **human-curated YouTube transcripts** (accurate text, segment-level timing)
- Enriches them with **WhisperX word-level timestamps** (precise timing data)
- Uses **AI-powered alignment** (Qwen3) to match words between sources
- Outputs **enriched transcripts** with both human accuracy and machine precision

## 🔧 Key Features

- **Hybrid Approach**: Combines human accuracy with machine precision
- **AI Alignment**: Uses Qwen3 for intelligent word matching with fallback to SequenceMatcher
- **Temporal Windowing**: Smart candidate filtering based on timing proximity
- **Contraction Handling**: Handles differences like "we're" vs "we 're" between sources
- **Quality Metrics**: Detailed statistics and confidence scoring
- **Debug Mode**: Comprehensive logging and diff reports for development

## 🏗️ The Problem TranscriptFusion Solves

**YouTube Official Transcripts**: Human-verified text but only segment-level timing  
**WhisperX Transcripts**: Word-level timing but potential transcription errors

**TranscriptFusion Result**: Human accuracy + Word-level timing = Perfect transcripts

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- WhisperX installed (`pip install whisperx`)
- yt-dlp for audio download (`pip install yt-dlp`)
- Optional: Local Qwen3 model for AI alignment

### Installation

```bash
git clone https://github.com/yourusername/TranscriptFusion.git
cd TranscriptFusion
pip install -r requirements.txt
```

### Basic Usage

1. **Download YouTube audio**:
   ```bash
   python data-processing/download_youtube_audio.py https://youtube.com/watch?v=VIDEO_ID
   ```

2. **Fetch official transcript**:
   ```bash
   python data-processing/fetch_official_transcript.py VIDEO_ID
   ```

3. **Generate WhisperX transcript**:
   ```bash
   python data-processing/transcribe_with_whisperx.py data/audio_clips/VIDEO_ID.mp3
   ```

4. **Enrich the transcript**:
   ```bash
   python data-processing/enrich_transcript.py VIDEO_ID
   ```

### Development Mode

For detailed debugging and analysis:

```bash
python data-processing/enrich_transcript.py VIDEO_ID --dev
```

This generates comprehensive debug reports in the `logs/` directory.

### Quick Test with Makefile

```bash
make setup && make install && make test-run
```

## 📁 Project Structure

```
TranscriptFusion/
├── data-processing/
│   ├── download_youtube_audio.py    # Download audio from YouTube
│   ├── fetch_official_transcript.py # Get human transcripts
│   ├── transcribe_with_whisperx.py  # Generate WhisperX transcripts
│   └── enrich_transcript.py         # Main enrichment pipeline
├── data/
│   ├── audio_clips/                 # Downloaded audio files
│   └── transcripts/                 # All transcript files
├── logs/                           # Debug reports (dev mode)
├── Makefile                        # Build and test automation
└── requirements.txt                # Python dependencies
```

## 🎬 Use Cases

- **Precise Subtitle Generation**: Create frame-perfect SRT/VTT files with accurate word boundaries
- **Speech Analysis**: Measure speaking pace, pauses, and word emphasis for research or coaching
- **Interactive Video**: Build clickable transcripts where users can jump to exact moments
- **Audio Editing**: Generate accurate cut points for podcast/video editing software
- **Language Learning Apps**: Highlight words in real-time as they're spoken for pronunciation training

## 🧠 AI Alignment

TranscriptFusion can use local Qwen3 for intelligent word alignment:

- Handles contractions (`"it's"` vs `"it 's"`)
- Manages compound words and hyphenation differences
- Provides confidence scoring for each alignment
- Falls back to SequenceMatcher if AI is unavailable

## 📊 Quality Metrics

The tool provides detailed statistics:
- Word matching success rate
- Confidence distribution
- Segment alignment quality
- Problematic word analysis

## 🚀 Next Steps

- **Modular LLM Support**: Make alignment model configurable via global settings file (Qwen variants, Phi-3, Llama, etc.)
- **Adaptive Windowing**: Dynamic buffer adjustment based on previous segment alignment quality
- **Performance Benchmarking**: Compare smaller models for speed vs accuracy trade-offs with automated metrics collection

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [WhisperX](https://github.com/m-bain/whisperX) for precise speech recognition
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) for transcript access
- [Qwen3](https://github.com/QwenLM/Qwen) for AI-powered alignment

---

**TranscriptFusion**: Where human accuracy meets machine precision ⚡