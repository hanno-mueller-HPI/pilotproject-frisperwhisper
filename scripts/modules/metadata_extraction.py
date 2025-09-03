#############################################################################
# Module: metadata_extraction.py                                            #
# Description: Extract metadata from TRS and TextGrid files                 #
# Author: Hanno Müller                                                      #
# Date: 2025-09-03                                                          #
#############################################################################

import os
import re
import chardet
import xml.etree.ElementTree as ET
from pathlib import Path


class TextGrid:
    """TextGrid parser for extracting segments with timing information."""
    
    def __init__(self, path, content):
        self.path = path
        self.content = content
        self.xmin = None
        self.xmax = None
        self.items = []  # List of dicts: {'name': ..., 'intervals': [...]}
        if content is not None:
            self._parse()

    @staticmethod
    def load_textgrid(path):
        """Load a TextGrid file with automatic encoding detection."""
        try:
            # Detect encoding
            with open(path, "rb") as f:
                rawdata = f.read()
            result = chardet.detect(rawdata)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'
        
            # Try detected encoding first
            try:
                with open(path, "r", encoding=encoding) as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Fallback encodings
                for fallback_encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        with open(path, "r", encoding=fallback_encoding) as f:
                            content = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # Last resort with error replacement
                    with open(path, "r", encoding='utf-8', errors='replace') as f:
                        content = f.read()
        
            return TextGrid(path=path, content=content)
        except Exception as e:
            print(f"Error loading TextGrid {os.path.basename(path)}: {e}")
            return None

    def _parse(self):
        """Parse TextGrid content to extract timing and speaker information."""
        lines = self.content.splitlines()

        # Extract global timing
        for line in lines:
            if line.strip().startswith("xmin ="):
                num_str = re.findall(r"[-+]?\d*\.\d+|\d+", line.strip().split("=")[1])[0]
                self.xmin = float(num_str)
                break
        for line in lines:
            if line.strip().startswith("xmax ="):
                num_str = re.findall(r"[-+]?\d*\.\d+|\d+", line.strip().split("=")[1])[0]
                self.xmax = float(num_str)
                break

        # Parse items (speakers)
        item_indices = [i for i, l in enumerate(lines) if l.strip().startswith("item [")]
        for idx, start in enumerate(item_indices):
            end = item_indices[idx + 1] if idx + 1 < len(item_indices) else len(lines)
            item_lines = lines[start:end]
            name = None
            intervals = []
            interval_block = []
            inside_interval = False
            
            for l in item_lines:
                stripped = l.strip()
                if stripped.startswith('name ='):
                    name = stripped.split('=')[1].strip().strip('"')
                if stripped.startswith('intervals ['):
                    inside_interval = True
                    interval_block = [stripped]
                elif inside_interval and (stripped.startswith('xmin =') or 
                                        stripped.startswith('xmax =') or 
                                        stripped.startswith('text =')):
                    interval_block.append(stripped)
                    if stripped.startswith('text ='):
                        # End of interval block
                        interval = {}
                        for entry in interval_block:
                            if entry.startswith('xmin ='):
                                interval['xmin'] = float(entry.split('=')[1])
                            elif entry.startswith('xmax ='):
                                interval['xmax'] = float(entry.split('=')[1])
                            elif entry.startswith('text ='):
                                interval['text'] = entry.split('=', 1)[1].strip().strip('"')
                        intervals.append(interval)
                        inside_interval = False
                        interval_block = []
                        
            if name and intervals:
                self.items.append({'name': name, 'intervals': intervals})

    def get_segments(self):
        """Extract all valid segments with metadata."""
        segments = []
        
        for item in self.items:
            speaker = item['name']
            for idx, interval in enumerate(item['intervals'], 1):
                # Only include intervals with valid time ranges and non-empty text
                if (interval['xmin'] < interval['xmax'] and 
                    interval['text'].strip()):
                    
                    segment = {
                        'speaker': speaker,
                        'start_time': interval['xmin'],
                        'end_time': interval['xmax'],
                        'duration': interval['xmax'] - interval['xmin'],
                        'text': interval['text'],
                        'segment_id': f"{speaker}_{idx}"
                    }
                    segments.append(segment)
        
        return segments


def extract_markers(text):
    """
    Extract markers (brackets, XXX patterns) from transcript text.
    
    Args:
        text (str): Original transcript text
        
    Returns:
        list: List of marker patterns found
    """
    markers = []
    
    # Bracket patterns
    bracket_patterns = [
        r'\([^)]*\)',  # (...)
        r'\[[^\]]*\]', # [...]
        r'<[^>]*>',    # <...>
    ]
    
    for pattern in bracket_patterns:
        matches = re.findall(pattern, text)
        markers.extend(matches)
    
    # XXX patterns (2+ consecutive X characters)
    xxx_matches = re.findall(r'X{2,}', text, re.IGNORECASE)
    markers.extend(xxx_matches)
    
    # Anonymous patterns
    anon_matches = re.findall(r'\b[Aa]non\b', text)
    markers.extend(anon_matches)
    
    return markers


def extract_segments_from_folder(data_dir, file_limit=None):
    """
    Extract all segments from TextGrid files in a folder.
    
    Args:
        data_dir (str): Path to directory containing TextGrid files
        file_limit (int, optional): Limit number of files for testing
        
    Returns:
        list: List of segment dictionaries with metadata
    """
    print(f"Extracting segments from {data_dir}...")
    
    # Find all TextGrid files
    textgrid_files = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".TextGrid"):
            textgrid_files.append(os.path.join(data_dir, filename))
    
    if file_limit:
        textgrid_files = textgrid_files[:file_limit]
        print(f"   Processing {len(textgrid_files)} files (limited for testing)")
    else:
        print(f"   Processing {len(textgrid_files)} TextGrid files")
    
    all_segments = []
    processed_files = 0
    
    for textgrid_path in textgrid_files:
        try:
            # Load TextGrid
            textgrid = TextGrid.load_textgrid(textgrid_path)
            if textgrid is None:
                continue
            
            # Get base filename
            filename = Path(textgrid_path).stem  # e.g., 'a018a'
            
            # Get speaker metadata from TRS file
            speaker_metadata = get_speaker_metadata(data_dir, filename)
            
            # Check if corresponding audio file exists
            audio_path = os.path.join(data_dir, filename + ".wav")
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file {filename}.wav not found")
                continue
            
            # Extract segments
            segments = textgrid.get_segments()
            
            # Add file-level metadata to each segment
            for segment in segments:
                speaker_id = segment.get('speaker', '')
                speaker_info = speaker_metadata.get(speaker_id, {})
                
                segment.update({
                    'filename': filename,
                    'textgrid_path': textgrid_path,
                    'audio_path': audio_path,
                    'markers': extract_markers(segment['text']),
                    'gender': speaker_info.get('gender', ''),
                    'dialect': speaker_info.get('dialect', ''),
                    'accent': speaker_info.get('accent', '')
                })
                all_segments.append(segment)
            
            processed_files += 1
            if processed_files % 10 == 0:
                print(f"   Processed {processed_files}/{len(textgrid_files)} files, "
                      f"{len(all_segments)} segments total")
                      
        except Exception as e:
            print(f"Warning: Error processing {os.path.basename(textgrid_path)}: {e}")
    
    print(f"Extraction completed:")
    print(f"   Files processed: {processed_files}")
    print(f"   Total segments: {len(all_segments)}")
    
    return all_segments


def get_audio_info(audio_path):
    """
    Get basic audio file information without loading the full file.
    
    Args:
        audio_path (str): Path to audio file
        
    Returns:
        dict: Audio metadata (sampling_rate, duration, channels)
    """
    try:
        import soundfile as sf
        with sf.SoundFile(audio_path) as audio_file:
            return {
                'sampling_rate': audio_file.samplerate,
                'duration': audio_file.frames / audio_file.samplerate,
                'channels': audio_file.channels
            }
    except Exception as e:
        print(f"Warning: Could not read audio info for {audio_path}: {e}")
        return {
            'sampling_rate': 16000,  # Default assumption
            'duration': 0.0,
            'channels': 1
        }


def parse_trs_file(trs_path):
    """
    Parse TRS file to extract speaker metadata including gender information.
    
    Args:
        trs_path (str): Path to TRS file
        
    Returns:
        dict: Speaker ID to metadata mapping
    """
    speaker_info = {}
    
    try:
        tree = ET.parse(trs_path)
        root = tree.getroot()
        
        # Find all speaker elements
        speakers = root.find('Speakers')
        if speakers is not None:
            for speaker in speakers.findall('Speaker'):
                speaker_id = speaker.get('id', '')
                speaker_name = speaker.get('name', '')
                speaker_gender = speaker.get('type', '')  # 'male' or 'female'
                dialect = speaker.get('dialect', '')
                accent = speaker.get('accent', '')
                
                speaker_info[speaker_id] = {
                    'name': speaker_name,
                    'gender': speaker_gender,
                    'dialect': dialect,
                    'accent': accent
                }
                
    except Exception as e:
        print(f"Warning: Could not parse TRS file {os.path.basename(trs_path)}: {e}")
    
    return speaker_info


def get_speaker_metadata(data_dir, filename):
    """
    Get speaker metadata from corresponding TRS file.
    
    Args:
        data_dir (str): Directory containing the files
        filename (str): Base filename (without extension)
        
    Returns:
        dict: Speaker ID to metadata mapping
    """
    # Look for TRS file in the same directory first
    trs_path = os.path.join(data_dir, filename + '.trs')
    if os.path.exists(trs_path):
        return parse_trs_file(trs_path)
    
    # Fallback: look in original data directory
    trs_path = os.path.join(data_dir, '..', 'data_original', filename + '.trs')
    if os.path.exists(trs_path):
        return parse_trs_file(trs_path)
    
    return {}
