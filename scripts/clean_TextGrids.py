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
    
    if not audio_array:
        logger.debug(f"Removing entry with no audio data")
        return True
    
    duration_seconds = len(audio_array) / sampling_rate
    
    if duration_seconds < min_duration_seconds:
        logger.debug(f"Removing entry with short duration: {duration_seconds:.3f}s (audio-based)")
        return True
    
    return False


def clean_textgrid_entries(entries, remove_buzz_anon=True, remove_empty=True, remove_short_audio=True, min_duration=0.1):
    """
    Apply all cleaning filters to a list of TextGrid entries.
    
    Args:
        entries (list): List of dataset entries
        remove_buzz_anon (bool): Whether to remove entries containing "(buzz) anon (buzz)"
        remove_empty (bool): Whether to remove empty or whitespace-only entries
        remove_short_audio (bool): Whether to remove very short audio segments
        min_duration (float): Minimum audio duration in seconds
        
    Returns:
        list: Filtered list of entries
    """
    if not entries:
        return entries
    
    original_count = len(entries)
    filtered_entries = []
    
    removed_counts = {
        'buzz_anon': 0,
        'empty': 0,
        'short_audio': 0
    }
    
    for entry in entries:
        should_remove = False
        
        # Check buzz anon pattern
        if remove_buzz_anon and should_remove_buzz_anon(entry):
            removed_counts['buzz_anon'] += 1
            should_remove = True
        
        # Check empty/whitespace
        if not should_remove and remove_empty and should_remove_empty_or_whitespace(entry):
            removed_counts['empty'] += 1
            should_remove = True
        
        # Check short audio
        if not should_remove and remove_short_audio and should_remove_very_short_audio(entry, min_duration):
            removed_counts['short_audio'] += 1
            should_remove = True
        
        if not should_remove:
            filtered_entries.append(entry)
    
    total_removed = sum(removed_counts.values())
    
    if total_removed > 0:
        logger.info(f"Cleaning results: {original_count} -> {len(filtered_entries)} entries "
                   f"(removed {total_removed}: {removed_counts['buzz_anon']} buzz_anon, "
                   f"{removed_counts['empty']} empty, {removed_counts['short_audio']} short_audio)")
    
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
