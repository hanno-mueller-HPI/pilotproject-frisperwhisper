#############################################################################
# Script Name: clean_TextGrids.py                                           #
# Description: Filtering functions to identify intervals that should be     #
#              excluded from TextGrid dataset processing                    #
# Author: Hanno Müller                                                      #
# Date: 2025-07-16                                                          #
#############################################################################

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def should_remove_buzz_anon(entry):
    """
    Check if an entry contains the string "(buzz) anon (buzz)" and should be removed.
    
    Args:
        entry (dict): Dataset entry with 'sentence' field
        
    Returns:
        bool: True if the entry should be removed, False otherwise
    """
    sentence = entry.get('sentence', '').strip()
    
    # Check if sentence contains "(buzz) anon (buzz)"
    if "(buzz) anon (buzz)" in sentence:
        logger.debug(f"Removing entry with buzz pattern: '{sentence}'")
        return True
    
    return False


def should_remove_buzz_patterns(entry):
    """
    Check if an entry contains any (buzz) patterns and should be removed.
    This includes (buzz), (buzz) text (buzz), etc.
    
    Args:
        entry (dict): Dataset entry with 'sentence' field
        
    Returns:
        bool: True if the entry should be removed, False otherwise
    """
    sentence = entry.get('sentence', '').strip()
    
    # Check if sentence contains any (buzz) pattern
    if "(buzz)" in sentence:
        logger.debug(f"Removing entry with buzz pattern: '{sentence}'")
        return True
    
    return False


def should_remove_brackets(entry):
    """
    Check if an entry contains any brackets ( ) [ ] < > and should be removed.
    
    Args:
        entry (dict): Dataset entry with 'sentence' field
        
    Returns:
        bool: True if the entry should be removed, False otherwise
    """
    sentence = entry.get('sentence', '').strip()
    
    # Check if sentence contains any of the specified brackets
    bracket_chars = ['(', ')', '[', ']', '<', '>']
    
    for bracket in bracket_chars:
        if bracket in sentence:
            logger.debug(f"Removing entry with bracket '{bracket}': '{sentence}'")
            return True
    
    return False


def should_remove_xxx_patterns(entry):
    """
    Check if an entry contains two or more consecutive X characters (case-insensitive) and should be removed.
    This includes patterns like XX, XXX, xxx, XXXX, fa le XXX, etc.
    
    Args:
        entry (dict): Dataset entry with 'sentence' field
        
    Returns:
        bool: True if the entry should be removed, False otherwise
    """
    sentence = entry.get('sentence', '').strip()
    
    # Check if sentence contains 2 or more consecutive X characters (case-insensitive)
    import re
    if re.search(r'[Xx]{2,}', sentence):
        logger.debug(f"Removing entry with XXX pattern: '{sentence}'")
        return True
    
    return False


def should_remove_files(entry, files_to_remove=None):
    """
    Check if an entry comes from specific files that should be completely excluded.
    
    Args:
        entry (dict): Dataset entry with potential 'path', 'textgrid_path', or 'audio_path' field
        files_to_remove (list or str): File basename(s) to remove (e.g., ['h015a', 'e025a'])
        
    Returns:
        bool: True if the entry should be removed, False otherwise
    """
    if files_to_remove is None:
        return False
    
    # Convert single file to list
    if isinstance(files_to_remove, str):
        files_to_remove = [files_to_remove]
    
    # Check various possible path field names
    path_fields = ['path', 'textgrid_path', 'audio_path']
    
    for field in path_fields:
        if field in entry:
            file_path = str(entry[field]).strip()
            
            # Extract basename without extension
            import os
            basename = os.path.splitext(os.path.basename(file_path))[0]
            
            # Check if current file should be removed
            for file_to_remove in files_to_remove:
                if basename.lower() == str(file_to_remove).strip().lower():
                    logger.debug(f"Removing entry from file: {basename}")
                    return True
    
    return False


def should_remove_empty_or_whitespace(entry):
    """
    Check if an entry contains only empty text or whitespace and should be removed.
    
    Args:
        entry (dict): Dataset entry with 'sentence' field
        
    Returns:
        bool: True if the entry should be removed, False otherwise
    """
    sentence = entry.get('sentence', '').strip()
    
    if not sentence:
        logger.debug(f"Removing entry with empty sentence")
        return True
    
    return False


def should_remove_very_short_audio(entry, min_duration_seconds=0.1):
    """
    Check if an entry has very short audio duration and should be removed.
    
    Args:
        entry (dict): Dataset entry with either:
                     - 'audio' field containing 'array' and 'sampling_rate' (processed entries)
                     - 'start_time', 'end_time' fields (lightweight entries)
        min_duration_seconds (float): Minimum duration in seconds
        
    Returns:
        bool: True if the entry should be removed, False otherwise
    """
    # Method 1: Check if entry has timing information (lightweight entries)
    if 'start_time' in entry and 'end_time' in entry:
        duration_seconds = entry['end_time'] - entry['start_time']
        if duration_seconds < min_duration_seconds:
            logger.debug(f"Removing entry with short duration: {duration_seconds:.3f}s (timing-based)")
            return True
        return False
    
    # Method 2: Check if entry has audio data (processed entries)
    audio_data = entry.get('audio', {})
    audio_array = audio_data.get('array', [])
    sampling_rate = audio_data.get('sampling_rate', 16000)
    
    # Check for completely missing or empty audio array
    if not audio_array or len(audio_array) == 0:
        logger.debug(f"Removing entry with empty audio array")
        return True
    
    duration_seconds = len(audio_array) / sampling_rate
    
    if duration_seconds < min_duration_seconds:
        logger.debug(f"Removing entry with short duration: {duration_seconds:.3f}s (audio-based)")
        return True
    
    return False


def should_remove_silent_audio(entry, silence_threshold=1e-6):
    """
    Check if an entry contains only silent audio (all zeros or near-zero values) and should be removed.
    
    Args:
        entry (dict): Dataset entry with 'audio' field containing 'array'
        silence_threshold (float): Threshold below which audio is considered silent
        
    Returns:
        bool: True if the entry should be removed, False otherwise
    """
    # Only check processed entries with audio data
    audio_data = entry.get('audio', {})
    audio_array = audio_data.get('array', [])
    
    if not audio_array or len(audio_array) == 0:
        # Empty audio arrays are handled by should_remove_very_short_audio
        return False
    
    try:
        import numpy as np
        
        # Convert to numpy array for efficient computation
        arr = np.array(audio_array)
        
        # Check if all values are below the silence threshold
        max_amplitude = np.max(np.abs(arr))
        
        if max_amplitude <= silence_threshold:
            logger.debug(f"Removing entry with silent audio (max amplitude: {max_amplitude:.8f})")
            return True
            
    except ImportError:
        # Fallback without numpy - check if all values are exactly zero
        max_abs_value = max(abs(x) for x in audio_array)
        if max_abs_value <= silence_threshold:
            logger.debug(f"Removing entry with silent audio (max amplitude: {max_abs_value:.8f})")
            return True
    except Exception as e:
        logger.warning(f"Error checking for silent audio: {e}")
        return False
    
    return False


def should_remove_speaker(entry, speakers_to_remove=None):
    """
    Check if an entry belongs to a specific speaker that should be removed.
    
    Args:
        entry (dict): Dataset entry with potential 'speaker', 'speaker_id', or 'speaker_name' field
        speakers_to_remove (list or str): Speaker(s) to remove (e.g., ['speaker1'] or 'speaker1')
        
    Returns:
        bool: True if the entry should be removed, False otherwise
    """
    if speakers_to_remove is None:
        return False
    
    # Convert single speaker to list
    if isinstance(speakers_to_remove, str):
        speakers_to_remove = [speakers_to_remove]
    
    # Check various possible speaker field names
    speaker_fields = ['speaker', 'speaker_id', 'speaker_name', 'spk', 'spk_id']
    
    for field in speaker_fields:
        if field in entry:
            speaker_value = str(entry[field]).strip().lower()
            
            # Check if current speaker should be removed
            for speaker_to_remove in speakers_to_remove:
                if speaker_value == str(speaker_to_remove).strip().lower():
                    logger.debug(f"Removing entry from speaker: {entry[field]}")
                    return True
    
    return False


def clean_textgrid_entries(entries, remove_buzz_anon=True, remove_buzz_patterns=True, remove_brackets=True,
                          remove_empty=True, remove_short_audio=True, remove_silent_audio=True, 
                          remove_speakers=None, remove_files=None, remove_xxx_patterns=True,
                          min_duration=0.1, silence_threshold=1e-6):
    """
    Apply all cleaning filters to a list of TextGrid entries.
    
    Args:
        entries (list): List of dataset entries
        remove_buzz_anon (bool): Whether to remove entries containing "(buzz) anon (buzz)"
        remove_buzz_patterns (bool): Whether to remove entries containing any "(buzz)" patterns
        remove_brackets (bool): Whether to remove entries containing brackets ( ) [ ] < >
        remove_empty (bool): Whether to remove empty or whitespace-only entries
        remove_short_audio (bool): Whether to remove very short audio segments
        remove_silent_audio (bool): Whether to remove silent audio (all zeros)
        remove_speakers (list or str): Speaker(s) to remove (e.g., ['speaker1'] or 'speaker1')
        remove_files (list or str): File basename(s) to remove completely (e.g., ['h015a', 'e025a'])
        remove_xxx_patterns (bool): Whether to remove entries containing 2+ consecutive X characters
        min_duration (float): Minimum audio duration in seconds
        silence_threshold (float): Threshold below which audio is considered silent
        
    Returns:
        list: Filtered list of entries
    """
    if not entries:
        return entries
    
    original_count = len(entries)
    filtered_entries = []
    
    removed_counts = {
        'buzz_anon': 0,
        'buzz_patterns': 0,
        'brackets': 0,
        'empty': 0,
        'short_audio': 0,
        'silent_audio': 0,
        'speaker_filter': 0,
        'file_filter': 0,
        'xxx_patterns': 0
    }
    
    for entry in entries:
        should_remove = False
        
        # Check file filter first (most specific)
        if remove_files and should_remove_files(entry, remove_files):
            removed_counts['file_filter'] += 1
            should_remove = True
        
        # Check speaker filter (specific)
        if not should_remove and remove_speakers and should_remove_speaker(entry, remove_speakers):
            removed_counts['speaker_filter'] += 1
            should_remove = True
        
        # Check buzz anon pattern (specific)
        if not should_remove and remove_buzz_anon and should_remove_buzz_anon(entry):
            removed_counts['buzz_anon'] += 1
            should_remove = True
        
        # Check general buzz patterns
        if not should_remove and remove_buzz_patterns and should_remove_buzz_patterns(entry):
            removed_counts['buzz_patterns'] += 1
            should_remove = True
        
        # Check brackets
        if not should_remove and remove_brackets and should_remove_brackets(entry):
            removed_counts['brackets'] += 1
            should_remove = True
        
        # Check XXX patterns (2+ consecutive X characters)
        if not should_remove and remove_xxx_patterns and should_remove_xxx_patterns(entry):
            removed_counts['xxx_patterns'] += 1
            should_remove = True
        
        # Check empty/whitespace
        if not should_remove and remove_empty and should_remove_empty_or_whitespace(entry):
            removed_counts['empty'] += 1
            should_remove = True
        
        # Check short audio (includes empty audio arrays)
        if not should_remove and remove_short_audio and should_remove_very_short_audio(entry, min_duration):
            removed_counts['short_audio'] += 1
            should_remove = True
        
        # Check silent audio (all zeros)
        if not should_remove and remove_silent_audio and should_remove_silent_audio(entry, silence_threshold):
            removed_counts['silent_audio'] += 1
            should_remove = True
        
        if not should_remove:
            filtered_entries.append(entry)
    
    total_removed = sum(removed_counts.values())
    
    if total_removed > 0:
        logger.info(f"Cleaning results: {original_count} -> {len(filtered_entries)} entries "
                   f"(removed {total_removed}: {removed_counts['buzz_anon']} buzz_anon, "
                   f"{removed_counts['buzz_patterns']} buzz_patterns, {removed_counts['brackets']} brackets, "
                   f"{removed_counts['xxx_patterns']} xxx_patterns, {removed_counts['empty']} empty, "
                   f"{removed_counts['short_audio']} short_audio, {removed_counts['silent_audio']} silent_audio, "
                   f"{removed_counts['speaker_filter']} speaker_filter, {removed_counts['file_filter']} file_filter)")
    
    return filtered_entries


def add_custom_filter(filter_function):
    """
    Decorator or function to add custom filtering functions.
    This allows for easy extension of the cleaning capabilities.
    
    Args:
        filter_function (callable): Function that takes an entry and returns True if it should be removed
    """
    # This is a placeholder for future extensibility
    # Custom filters can be added to the clean_textgrid_entries function
    pass
