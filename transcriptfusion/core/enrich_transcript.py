#!/usr/bin/env python3
"""
Transcript Timing Enrichment Tool for ArticuLoop Project
Attaches precise word-level timings from WhisperX to YouTube official transcripts.

Usage:
    python enrich_transcript.py <video_id>
    python enrich_transcript.py QkpqBCaUvS4 --dev
    
Expected files in data/transcripts/:
    - {video_id}_official.json (YouTube transcript)
    - {video_id}.json (WhisperX transcript)
    
Output:
    - {video_id}_enriched.json (Enhanced transcript)
"""

import sys
import os
import json
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from datetime import datetime
import requests


@dataclass
class WordTiming:
    """Individual word with precise timing"""
    word: str
    start: float
    end: float
    score: float = 0.0


@dataclass
class MatchResult:
    """Detailed result of word matching for debugging"""
    youtube_word: str
    youtube_idx: int
    whisper_word: str = None
    whisper_idx: int = -1
    confidence: float = 0.0
    timing_source: str = "interpolated"  # "matched" or "interpolated"
    start_time: float = 0.0
    end_time: float = 0.0
    segment_idx: int = 0
    notes: str = ""


@dataclass
class SegmentDebugInfo:
    """Debug information for a segment"""
    segment_idx: int
    youtube_text: str
    youtube_start: float
    youtube_end: float
    youtube_words: List[str]
    whisper_candidates: List[str]
    matches: List[MatchResult]
    alignment_quality: str  # "excellent", "good", "poor", "failed"


@dataclass
class EnrichedSegment:
    """YouTube segment enriched with word-level timings"""
    start: float
    end: float
    text: str
    words: List[WordTiming]


class ArticuLoopTimingEnricher:
    def __init__(self, project_root: str = None, dev_mode: bool = False, use_qwen: bool = True):
        if project_root is None:
            # Auto-detect project root from script location
            self.project_root = os.getcwd()
        else:
            self.project_root = project_root
            
        self.transcripts_dir = os.path.join(self.project_root, "data", "transcripts")
        self.logs_dir = os.path.join(self.project_root, "logs")
        self.word_pattern = re.compile(r'\b\w+\b|\S')  # Words and punctuation
        self.dev_mode = dev_mode
        self.use_qwen = use_qwen
        self.debug_data = []  # Store all debug information
        
        # Create logs directory if in dev mode
        if self.dev_mode:
            os.makedirs(self.logs_dir, exist_ok=True)
            print(f"🐛 Development mode enabled - logs will be saved to: {self.logs_dir}")
        
        print(f"📁 Project root: {self.project_root}")
        print(f"📁 Transcripts directory: {self.transcripts_dir}")
        if self.use_qwen:
            print(f"🧠 Using Qwen3 for alignment")
        
    def normalize_word(self, word: str) -> str:
        """Normalize word for matching (lowercase, strip punctuation, handle numbers)"""
        # First normalize numbers
        normalized = self.normalize_numbers(word)
        # Then apply existing normalization
        normalized = re.sub(r'[^\w]', '', normalized.lower())
        return normalized

    def analyze_contractions(self, youtube_segments: List[Dict], whisper_words: List[WordTiming]):
        """Analyze how contractions appear in both transcripts"""
        print("\n🔍 CONTRACTION ANALYSIS:")
        
        # Find contractions in both sources
        youtube_contractions = []
        whisper_contractions = []
        
        for segment in youtube_segments:
            tokens = self.tokenize_text(segment['text'])
            for token in tokens:
                if "'" in token:
                    youtube_contractions.append(token)
        
        for word in whisper_words:
            if "'" in word.word:
                whisper_contractions.append(word.word)
        
        print(f"YouTube contractions: {set(youtube_contractions)}")
        print(f"WhisperX contractions: {set(whisper_contractions)}")
        
        # Look for split patterns
        youtube_splits = []
        for segment in youtube_segments:
            tokens = self.tokenize_text(segment['text'])
            for i, token in enumerate(tokens[:-1]):
                if tokens[i+1].startswith("'"):
                    youtube_splits.append(f"{token} + {tokens[i+1]}")
        
        print(f"YouTube split patterns: {set(youtube_splits)}")
            
    def preprocess_contractions(self, words: List[str]) -> List[str]:
        """Preprocess contractions to handle YouTube vs WhisperX differences"""
        processed = []
        i = 0
        
        while i < len(words):
            # Check if current word + next word forms a contraction
            if i < len(words) - 1 and words[i+1].startswith("'"):
                # Combine split contraction: "we" + "'re" → "we're"
                combined = words[i] + words[i+1]
                processed.append(combined)
                i += 2  # Skip both parts
            else:
                processed.append(words[i])
                i += 1
        
        return processed
        
    def tokenize_text(self, text: str) -> List[str]:
        """Split text into words while preserving original spacing/punctuation context"""
        return self.word_pattern.findall(text)
    
    def load_whisperx_transcript(self, video_id: str) -> List[WordTiming]:
        """Load WhisperX transcript and extract all word timings"""
        whisperx_path = os.path.join(self.transcripts_dir, f"{video_id}.json")
        
        if not os.path.exists(whisperx_path):
            print(f"❌ WhisperX file not found: {whisperx_path}")
            return []
        
        try:
            with open(whisperx_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            word_timings = []
            
            # Use word_segments if available (flat list), otherwise extract from segments
            if 'word_segments' in data:
                print("📄 Using word_segments from WhisperX")
                for word_data in data['word_segments']:
                    word_timings.append(WordTiming(
                        word=word_data['word'].strip(),
                        start=word_data['start'],
                        end=word_data['end'],
                        score=word_data.get('score', 0.0)
                    ))
            else:
                print("📄 Extracting words from segments")
                # Extract from segments.words
                for segment in data.get('segments', []):
                    for word_data in segment.get('words', []):
                        word_timings.append(WordTiming(
                            word=word_data['word'].strip(),
                            start=word_data['start'],
                            end=word_data['end'],
                            score=word_data.get('score', 0.0)
                        ))
            
            print(f"✅ Loaded {len(word_timings)} word timings from WhisperX")
            return word_timings
            
        except Exception as e:
            print(f"❌ Error loading WhisperX transcript: {e}")
            return []

    def load_youtube_transcript(self, video_id: str) -> List[Dict]:
        """Load YouTube official transcript"""
        youtube_path = os.path.join(self.transcripts_dir, f"{video_id}_official.json")
        
        if not os.path.exists(youtube_path):
            print(f"❌ YouTube official file not found: {youtube_path}")
            return []
        
        try:
            with open(youtube_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            segments = []
            for seg in data:
                segments.append({
                    'start': seg['start'],
                    'end': seg['start'] + seg['duration'],
                    'text': seg['text'].strip()
                })
            
            print(f"✅ Loaded {len(segments)} YouTube segments")
            return segments
            
        except Exception as e:
            print(f"❌ Error loading YouTube transcript: {e}")
            return []

    def find_temporal_window(self, youtube_start: float, youtube_end: float, 
                        whisper_words: List[WordTiming], 
                        buffer: float = 3.0) -> List[WordTiming]:
        """Find WhisperX words that are likely to be in the given time window"""
        window_start = max(0, youtube_start - buffer)
        window_end = youtube_end + buffer
        
        window_words = []
        for word in whisper_words:
            # Word overlaps with window if its midpoint or boundaries are within range
            word_mid = (word.start + word.end) / 2
            if (window_start <= word_mid <= window_end or 
                window_start <= word.start <= window_end or 
                window_start <= word.end <= window_end):
                window_words.append(word)
        
        return window_words

    def call_qwen3(self, prompt: str) -> str:
        """Call Qwen3 locally and get the response"""
        try:
            # Use the local endpoint at 127.0.0.1:1234
            response = requests.post(
                "http://127.0.0.1:1234/v1/chat/completions", 
                json={
                    "model": "qwen/qwen3-14b",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{prompt} /nothink"
                        }
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.7
                },
                timeout=90
            )
                        
            if response.status_code == 200:
                print('GOT 200!')
                print('response', response)
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"Error calling Qwen3: {response.status_code}")
                print(f"Response: {response.text}")
                return f"Error: {response.status_code}"
        except Exception as e:
            print(f"Exception calling Qwen3: {e}")
            return f"Exception: {str(e)}"

    def align_using_qwen3(self, youtube_segment: Dict, whisper_words: List[WordTiming]) -> List[Tuple[int, int, float]]:
        """Use Qwen3 to align YouTube transcript text with WhisperX words"""
        
        youtube_text = youtube_segment['text']
        youtube_tokens = self.tokenize_text(youtube_text)
        youtube_words = [token for token in youtube_tokens if re.search(r'\w', token)]
        youtube_words = self.preprocess_contractions(youtube_words)  # Add this line
        
        whisper_candidates = [w.word for w in whisper_words]
        
        # Format prompt for Qwen3
        prompt = f"""
        I need to match words from one transcript to another for precise alignment.

        YOUTUBE TRANSCRIPT: "{youtube_text}"
        YOUTUBE WORDS (tokenized): {youtube_words}
        
        WHISPERX WORDS: {whisper_candidates}
        
        For each word in the YouTube transcript, find the best matching word from WhisperX.
        Pay special attention to contractions (like "it's", "we're", "don't") which might 
        appear as separate tokens ("it 's", "we 're", "don 't") in WhisperX.
        
        WhisperX has spaced contractions like "we 're" instead of "we're", so make sure to match these correctly.
        Also watch for compound words like "go-to" which might be split in one transcript but not the other.
        
        Return the result as a JSON list of objects with this structure:
        [
        {{
            "youtube_index": 0,  // Index in the YouTube words list
            "whisperx_index": 3,  // Index in the WhisperX words list
            "confidence": 0.95   // How confident you are in this match (0.0-1.0)
        }},
        ...
        ]
        
        If a YouTube word has no good match, omit it from the result. ANSWER WITH NOTHING BUT THE JSON RESPONSE.
        """
        
        # Call Qwen3
        response = self.call_qwen3(prompt)
        cleaned_result = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

        print('>>>now cleaned: \n', cleaned_result)
        print('\n end of cleaned <<<')
        
        # Parse the JSON response
        try:
            alignments = json.loads(cleaned_result)  # Use cleaned_result consistently
            
            # Convert to the format expected by our code
            result = []
            for alignment in alignments:
                youtube_idx = alignment.get("youtube_index", -1)
                whisperx_idx = alignment.get("whisperx_index", -1)
                confidence = alignment.get("confidence", 0.0)
                
                if (youtube_idx >= 0 and youtube_idx < len(youtube_words) and 
                    whisperx_idx >= 0 and whisperx_idx < len(whisper_words)):
                    result.append((youtube_idx, whisperx_idx, confidence))
            
            return result
        except Exception as e:
            print(f"Error parsing Qwen3 response: {e}")
            print(f"Response content: {cleaned_result}")  # Use cleaned_result for consistency
            return []
    
    def fallback_align_word_sequences(self, youtube_words: List[str], 
                           whisper_words: List[WordTiming]) -> List[Tuple[int, int, float]]:
        """Fallback alignment method using SequenceMatcher if Qwen3 fails"""
        # Create normalized word lists for comparison
        yt_normalized = [self.normalize_word(w) for w in youtube_words]
        wx_normalized = [self.normalize_word(w.word) for w in whisper_words]
        
        # Use SequenceMatcher to find aligned subsequences
        matcher = SequenceMatcher(None, yt_normalized, wx_normalized)
        matches = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # Perfect matches - align one-to-one
                for k in range(i2 - i1):
                    matches.append((i1 + k, j1 + k, 1.0))
            
            elif tag == 'replace' and (i2 - i1) == (j2 - j1):
                # Same length replacement - align with similarity scores
                for k in range(i2 - i1):
                    yt_word = yt_normalized[i1 + k]
                    wx_word = wx_normalized[j1 + k]
                    similarity = SequenceMatcher(None, yt_word, wx_word).ratio()
                    if similarity > 0.6:  # Threshold for fuzzy matches
                        matches.append((i1 + k, j1 + k, similarity))
        
        return matches
    
    def interpolate_timing(self, segment_start: float, segment_end: float, 
                          word_idx: int, total_words: int) -> Tuple[float, float]:
        """Interpolate timing for words without WhisperX matches"""
        if total_words <= 1:
            return segment_start, segment_end
        
        segment_duration = segment_end - segment_start
        word_duration = segment_duration / total_words
        
        word_start = segment_start + (word_idx * word_duration)
        word_end = word_start + word_duration
        
        return word_start, word_end
    
    def attach_timings_to_segment(self, youtube_segment: Dict, 
                                whisper_words: List[WordTiming], 
                                segment_idx: int = 0) -> Tuple[EnrichedSegment, Optional[SegmentDebugInfo]]:
        """Attach word-level timings to a single YouTube segment"""
        
        # Tokenize YouTube text into words
        youtube_tokens = self.tokenize_text(youtube_segment['text'])
        
        # Filter to actual words (skip pure punctuation)
        youtube_words = [token for token in youtube_tokens if re.search(r'\w', token)]
        
        debug_info = None
        if self.dev_mode:
            debug_info = SegmentDebugInfo(
                segment_idx=segment_idx,
                youtube_text=youtube_segment['text'],
                youtube_start=youtube_segment['start'],
                youtube_end=youtube_segment['end'],
                youtube_words=youtube_words.copy(),
                whisper_candidates=[],
                matches=[],
                alignment_quality="failed"
            )
        
        if not youtube_words:
            empty_segment = EnrichedSegment(
                start=youtube_segment['start'],
                end=youtube_segment['end'],
                text=youtube_segment['text'],
                words=[]
            )
            if debug_info:
                debug_info.alignment_quality = "empty"
            return empty_segment, debug_info
        
        # Find WhisperX words in temporal vicinity
        temporal_window = self.find_temporal_window(
            youtube_segment['start'], 
            youtube_segment['end'], 
            whisper_words
        )
        
        if self.dev_mode and debug_info:
            debug_info.whisper_candidates = [w.word for w in temporal_window]
        
        # Align word sequences using Qwen3 or fallback to SequenceMatcher
        if self.use_qwen:
            alignments = self.align_using_qwen3(youtube_segment, temporal_window)
            # If Qwen3 fails, fall back to SequenceMatcher
            if not alignments:
                print(f"⚠️ Qwen3 alignment failed for segment {segment_idx}, falling back to SequenceMatcher")
                alignments = self.fallback_align_word_sequences(youtube_words, temporal_window)
        else:
            alignments = self.fallback_align_word_sequences(youtube_words, temporal_window)
        
        # Create timing dictionary from alignments
        timing_map = {}
        high_confidence_matches = 0
        for yt_idx, wx_idx, confidence in alignments:
            if confidence > 0.6:  # Only use high-confidence matches
                timing_map[yt_idx] = temporal_window[wx_idx]
                if confidence > 0.8:
                    high_confidence_matches += 1
        
        # Determine alignment quality for debugging
        if self.dev_mode and debug_info:
            match_ratio = len(timing_map) / len(youtube_words) if youtube_words else 0
            if match_ratio > 0.8:
                debug_info.alignment_quality = "excellent"
            elif match_ratio > 0.6:
                debug_info.alignment_quality = "good" 
            elif match_ratio > 0.3:
                debug_info.alignment_quality = "poor"
            else:
                debug_info.alignment_quality = "failed"
        
        # Build enriched word list
        enriched_words = []
        for i, word in enumerate(youtube_words):
            match_result = None
            
            if i in timing_map:
                # Use matched WhisperX timing
                wx_word = timing_map[i]
                enriched_words.append(WordTiming(
                    word=word,
                    start=wx_word.start,
                    end=wx_word.end,
                    score=wx_word.score
                ))
                
                if self.dev_mode:
                    match_result = MatchResult(
                        youtube_word=word,
                        youtube_idx=i,
                        whisper_word=wx_word.word,
                        whisper_idx=-1,  # Could track this if needed
                        confidence=wx_word.score,
                        timing_source="matched",
                        start_time=wx_word.start,
                        end_time=wx_word.end,
                        segment_idx=segment_idx,
                        notes=f"Direct match (score: {wx_word.score:.2f})"
                    )
            else:
                # Interpolate timing within segment
                word_start, word_end = self.interpolate_timing(
                    youtube_segment['start'],
                    youtube_segment['end'],
                    i,
                    len(youtube_words)
                )
                enriched_words.append(WordTiming(
                    word=word,
                    start=word_start,
                    end=word_end,
                    score=0.5  # Lower confidence for interpolated timings
                ))
                
                if self.dev_mode:
                    # Try to find closest whisper word for debugging
                    closest_word = ""
                    closest_similarity = 0.0
                    if temporal_window:
                        for wx_word in temporal_window:
                            sim = SequenceMatcher(None, self.normalize_word(word), 
                                                self.normalize_word(wx_word.word)).ratio()
                            if sim > closest_similarity:
                                closest_similarity = sim
                                closest_word = wx_word.word
                    
                    match_result = MatchResult(
                        youtube_word=word,
                        youtube_idx=i,
                        whisper_word=closest_word,
                        whisper_idx=-1,
                        confidence=0.5,
                        timing_source="interpolated",
                        start_time=word_start,
                        end_time=word_end,
                        segment_idx=segment_idx,
                        notes=f"No match found, interpolated. Closest WhisperX: '{closest_word}' (sim: {closest_similarity:.2f})"
                    )
            
            if self.dev_mode and debug_info and match_result:
                debug_info.matches.append(match_result)
        
        enriched_segment = EnrichedSegment(
            start=youtube_segment['start'],
            end=youtube_segment['end'],
            text=youtube_segment['text'],
            words=enriched_words
        )
        
        return enriched_segment, debug_info

    def normalize_numbers(self, word: str) -> str:
        """Convert between numeric and word forms"""
        # Number to word mapping
        num_to_word = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
            '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
            '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
            '18': 'eighteen', '19': 'nineteen', '20': 'twenty', '30': 'thirty',
            '40': 'forty', '50': 'fifty', '60': 'sixty', '70': 'seventy',
            '80': 'eighty', '90': 'ninety', '100': 'hundred', '1000': 'thousand'
        }
        
        # Create reverse mapping
        word_to_num = {v: k for k, v in num_to_word.items()}
        
        # Check if word is a number that can be converted to word form
        if word in num_to_word:
            return num_to_word[word]
        
        # Check if word is a word form that can be converted to number
        if word.lower() in word_to_num:
            return word_to_num[word.lower()]
        
        return word

    def enrich_transcript(self, youtube_segments: List[Dict], 
                         whisper_words: List[WordTiming]) -> List[EnrichedSegment]:
        """Main enrichment process: attach timings to all YouTube segments"""
        
        enriched_segments = []
        total_segments = len(youtube_segments)
        
        print("🔗 Attaching word-level timings to YouTube segments...")
        
        for i, yt_segment in enumerate(youtube_segments):
            # Progress indicator
            if i % 10 == 0 or i == total_segments - 1:
                print(f"   Processing segment {i+1}/{total_segments}")
            
            enriched_segment, debug_info = self.attach_timings_to_segment(
                yt_segment, whisper_words, segment_idx=i
            )
            enriched_segments.append(enriched_segment)
            
            # Store debug information
            if self.dev_mode and debug_info:
                self.debug_data.append(debug_info)
        
        print(f"✅ Completed enrichment of {len(enriched_segments)} segments")
        return enriched_segments

    def calculate_statistics(self, enriched_segments: List[EnrichedSegment]):
        """Calculate and display enrichment statistics"""
        total_words = 0
        matched_words = 0
        interpolated_words = 0
        high_confidence_words = 0
        
        for segment in enriched_segments:
            for word in segment.words:
                total_words += 1
                if word.score > 0.7:
                    matched_words += 1
                    if word.score > 0.8:
                        high_confidence_words += 1
                else:
                    interpolated_words += 1
        
        print(f"\n📊 Enrichment Statistics:")
        print(f"   Total words: {total_words}")
        print(f"   Matched with WhisperX: {matched_words} ({matched_words/total_words*100:.1f}%)")
        print(f"   High confidence matches: {high_confidence_words} ({high_confidence_words/total_words*100:.1f}%)")
        print(f"   Interpolated timings: {interpolated_words} ({interpolated_words/total_words*100:.1f}%)")

    def save_enriched_transcript(self, enriched_segments: List[EnrichedSegment], 
                               video_id: str):
        """Save enriched transcript in WhisperX-compatible format"""
        
        segments_output = []
        word_segments_output = []
        
        for segment in enriched_segments:
            # Convert words to dict format
            words_dict = []
            for word in segment.words:
                word_dict = {
                    'word': word.word,
                    'start': word.start,
                    'end': word.end,
                    'score': word.score
                }
                words_dict.append(word_dict)
                word_segments_output.append(word_dict)
            
            # Add segment
            segments_output.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
                'words': words_dict
            })
        
        # Create output structure
        output_data = {
            'segments': segments_output,
            'word_segments': word_segments_output,
            'language': 'en'  # Default language
        }
        
        # Save to enriched file
        output_path = os.path.join(self.transcripts_dir, f"{video_id}_enriched.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Enriched transcript saved to: {output_path}")

    def generate_debug_reports(self, video_id: str):
        """Generate comprehensive debug reports for development"""
        if not self.dev_mode or not self.debug_data:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate diff-style report
        self._generate_diff_report(video_id, timestamp)
        
        # Generate statistics report  
        self._generate_stats_report(video_id, timestamp)
        
        print(f"🐛 Debug reports generated in {self.logs_dir}/")

    def _generate_diff_report(self, video_id: str, timestamp: str):
        """Generate a diff-style report showing YouTube vs WhisperX word matches"""
        diff_path = os.path.join(self.logs_dir, f"{video_id}_diff_{timestamp}.txt")
        
        with open(diff_path, 'w', encoding='utf-8') as f:
            f.write(f"TRANSCRIPT ENRICHMENT DIFF REPORT\n")
            f.write(f"Video ID: {video_id}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            
            # Add contraction detection report
            f.write("CONTRACTION ANALYSIS\n")
            f.write("-" * 50 + "\n")
            youtube_contractions = []
            whisperx_contractions = []
            
            for segment_debug in self.debug_data:
                for word in segment_debug.youtube_words:
                    if any(c in word for c in ["'s", "'t", "'re", "'ve", "'ll", "'d", "'m"]):
                        youtube_contractions.append(word)
                
                for word in segment_debug.whisper_candidates:
                    if " " in word and any(word.endswith(c) for c in ["s", "t", "re", "ve", "ll", "d", "m"]):
                        whisperx_contractions.append(word)
            
            f.write(f"YouTube contractions found: {len(youtube_contractions)}\n")
            f.write(f"Example YouTube contractions: {youtube_contractions[:10]}\n\n")
            f.write(f"WhisperX contractions found: {len(whisperx_contractions)}\n")
            f.write(f"Example WhisperX contractions: {whisperx_contractions[:10]}\n\n")
            
            # Regular diff report
            for segment_debug in self.debug_data:
                f.write(f"SEGMENT {segment_debug.segment_idx} [{segment_debug.youtube_start:.2f}s - {segment_debug.youtube_end:.2f}s]\n")
                f.write(f"Quality: {segment_debug.alignment_quality.upper()}\n")
                f.write(f"Text: \"{segment_debug.youtube_text}\"\n")
                f.write(f"WhisperX candidates: {segment_debug.whisper_candidates}\n")
                f.write("-" * 50 + "\n")
                
                for match in segment_debug.matches:
                    if match.timing_source == "matched":
                        f.write(f"✓ MATCH   : '{match.youtube_word}' → '{match.whisper_word}' (conf: {match.confidence:.2f})\n")
                    else:
                        f.write(f"✗ INTERP  : '{match.youtube_word}' → INTERPOLATED ({match.notes})\n")
                
                f.write("\n")
        
        print(f"   📝 Diff report: {diff_path}")

    def _generate_stats_report(self, video_id: str, timestamp: str):
        """Generate statistics and quality metrics report"""
        stats_path = os.path.join(self.logs_dir, f"{video_id}_stats_{timestamp}.txt")
        
        # Calculate overall statistics
        total_segments = len(self.debug_data)
        quality_counts = {"excellent": 0, "good": 0, "poor": 0, "failed": 0, "empty": 0}
        total_words = 0
        matched_words = 0
        interpolated_words = 0
        
        for segment_debug in self.debug_data:
            quality_counts[segment_debug.alignment_quality] += 1
            for match in segment_debug.matches:
                total_words += 1
                if match.timing_source == "matched":
                    matched_words += 1
                else:
                    interpolated_words += 1
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"ENRICHMENT QUALITY STATISTICS\n")
            f.write(f"Video ID: {video_id}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("SEGMENT QUALITY BREAKDOWN:\n")
            for quality, count in quality_counts.items():
                percentage = (count / total_segments * 100) if total_segments > 0 else 0
                f.write(f"  {quality.capitalize():>10}: {count:>3} ({percentage:>5.1f}%)\n")
            
            f.write(f"\nWORD MATCHING STATISTICS:\n")
            f.write(f"  Total words: {total_words}\n")
            if total_words > 0:
                f.write(f"  Matched: {matched_words} ({matched_words/total_words*100:.1f}%)\n")
                f.write(f"  Interpolated: {interpolated_words} ({interpolated_words/total_words*100:.1f}%)\n")
            
            # Top problem words (most frequently interpolated)
            problem_words = {}
            for segment_debug in self.debug_data:
                for match in segment_debug.matches:
                    if match.timing_source == "interpolated":
                        word = match.youtube_word.lower()
                        problem_words[word] = problem_words.get(word, 0) + 1
            
            if problem_words:
                f.write(f"\nTOP PROBLEMATIC WORDS (most often interpolated):\n")
                sorted_problems = sorted(problem_words.items(), key=lambda x: x[1], reverse=True)
                for word, count in sorted_problems[:10]:
                    f.write(f"  '{word}': {count} times\n")
        
        print(f"   📊 Stats report: {stats_path}")

    # Calculate and display statistics
    def process_video(self, video_id: str):
            """Main processing pipeline for a single video"""
            print(f"🎬 Processing video: {video_id}")
            print("=" * 50)
            
            # Load transcripts
            whisper_words = self.load_whisperx_transcript(video_id)
            youtube_segments = self.load_youtube_transcript(video_id)
            
            if not whisper_words:
                print(f"❌ Could not load WhisperX transcript for {video_id}")
                return False
            
            if not youtube_segments:
                print(f"❌ Could not load YouTube transcript for {video_id}")
                return False

            if self.dev_mode:
                self.analyze_contractions(youtube_segments, whisper_words)
            
            # Enrich YouTube transcript with WhisperX timings
            enriched_segments = self.enrich_transcript(youtube_segments, whisper_words)
            
            # Calculate and display statistics
            self.calculate_statistics(enriched_segments)
            
            # Save results
            self.save_enriched_transcript(enriched_segments, video_id)
            
            # Generate debug reports if in dev mode
            if self.dev_mode:
                self.generate_debug_reports(video_id)
            
            print(f"🎉 Successfully enriched transcript for {video_id}")
            return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Attach WhisperX word-level timings to YouTube transcript text"
    )
    parser.add_argument('video_id', help='Video ID to process')
    parser.add_argument('--dev', action='store_true', 
                       help='Enable development mode with detailed logging and diff reports')
    parser.add_argument('--no-qwen', action='store_true',
                       help='Disable Qwen3 for alignment and use fallback method only')
    
    args = parser.parse_args()
    
    if not args.video_id:
        print("Usage: python enrich_transcript.py <video_id> [--dev] [--no-qwen]")
        print("Example: python enrich_transcript.py QkpqBCaUvS4 --dev")
        print()
        print("Expected files in data/transcripts/:")
        print("  - {video_id}_official.json (YouTube transcript)")  
        print("  - {video_id}.json (WhisperX transcript)")
        print()
        print("Output:")
        print("  - {video_id}_enriched.json (Enhanced transcript)")
        print("  - logs/{video_id}_*_{timestamp}.txt (Debug reports, if --dev used)")
        sys.exit(1)
    
    video_id = args.video_id
    
    # Initialize enricher with dev mode
    enricher = ArticuLoopTimingEnricher(
        dev_mode=args.dev,
        use_qwen=not args.no_qwen
    )
    
    # Process the video
    success = enricher.process_video(video_id)
    
    if success:
        print("\n✨ Transcript enrichment completed successfully!")
        if args.dev:
            print("📋 Check the logs/ directory for detailed debugging reports!")
    else:
        print("\n💥 Transcript enrichment failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()