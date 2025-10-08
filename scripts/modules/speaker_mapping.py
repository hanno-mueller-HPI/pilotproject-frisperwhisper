#############################################################################
# Module: speaker_mapping.py                                                #
# Description: Handle multi-interview speaker identification and mapping    #
# Author: Hanno Müller                                                      #
# Date: 2025-09-03                                                          #
#############################################################################

import xml.etree.ElementTree as ET
import os
from collections import defaultdict


def create_speaker_mapping(data_dir):
    """
    Create mapping for speakers who appear in multiple interviews.
    
    Args:
        data_dir (str): Path to directory containing TRS files
        
    Returns:
        dict: Mapping from filename_speaker to consistent speaker_id
    """
    print("Analyzing speaker patterns across interviews...")
    
    # Track speakers across files
    speaker_appearances = defaultdict(list)  # speaker_name -> [files]
    speaker_details = defaultdict(dict)      # speaker_name -> {metadata}
    
    interviewer_patterns = ['999', 'a999', 'b999', 'c999', 'd999', 'e999', 'g999', 'h999']
    
    # Get all TRS files
    trs_files = [f for f in os.listdir(data_dir) if f.endswith('.trs')]
    
    if not trs_files:
        print("   No TRS files found - using TextGrid-only mode")
        return create_speaker_mapping_from_textgrids(data_dir)
    
    for trs_file in trs_files:
        try:
            tree = ET.parse(os.path.join(data_dir, trs_file))
            root = tree.getroot()
            
            # Get all speakers in this file
            speakers = root.findall('.//Speaker')
            for speaker in speakers:
                spk_name = speaker.get('name', '')
                spk_id = speaker.get('id', '')
                spk_type = speaker.get('type', '')
                spk_dialect = speaker.get('dialect', '')
                
                # Store all speakers (including interviewers)
                if spk_name:
                    speaker_appearances[spk_name].append(trs_file)
                    
                    # Store speaker details
                    if spk_name not in speaker_details:
                        speaker_details[spk_name] = {
                            'type': spk_type,
                            'dialect': spk_dialect,
                            'id': spk_id
                        }
                        
        except Exception as e:
                        print(f"Warning: Error parsing {trs_file}: {e}")    # Create consistent speaker mapping
    speaker_mapping = {}
    multi_interview_count = 0
    
    for speaker_name, files in speaker_appearances.items():
        details = speaker_details[speaker_name]
        
        # For interviewers, use consistent ID
        if details['dialect'] == 'nonnative' or speaker_name in interviewer_patterns:
            consistent_id = "interviewer_999"
        else:
            # For native speakers, use the speaker name as consistent ID
            consistent_id = speaker_name
            
        # Track multi-interview speakers
        if len(files) > 1:
            multi_interview_count += 1
            
        # Create mapping for each file where this speaker appears
        for trs_file in files:
            filename = os.path.splitext(trs_file)[0]  # Remove .trs extension
            key = f"{filename}_{speaker_name}"
            speaker_mapping[key] = {
                'speaker_id': consistent_id,
                'interview_number': filename,
                'gender': details['type'],
                'dialect': details['dialect'],
                'is_multi_interview': len(files) > 1
            }
    
    print(f"Speaker mapping created:")
    print(f"   Total speakers: {len(speaker_appearances)}")
    print(f"   Multi-interview speakers: {multi_interview_count}")
    print(f"   Total mappings: {len(speaker_mapping)}")
    
    return speaker_mapping


def create_speaker_mapping_from_textgrids(data_dir):
    """
    Create basic speaker mapping from TextGrid files when TRS files are not available.
    
    Args:
        data_dir (str): Path to directory containing TextGrid files
        
    Returns:
        dict: Basic mapping from filename_speaker to speaker info
    """
    print("   Creating basic speaker mapping from TextGrid files...")
    
    import glob
    
    # Find all TextGrid files
    textgrid_files = glob.glob(os.path.join(data_dir, "*.TextGrid"))
    
    speaker_mapping = {}
    
    for textgrid_path in textgrid_files:
        filename = os.path.splitext(os.path.basename(textgrid_path))[0]
        
        try:
            # Simple TextGrid parsing to extract speaker names
            with open(textgrid_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Extract speaker names from TextGrid structure
            import re
            speaker_names = set()
            
            # Look for tier names (speakers)
            name_matches = re.findall(r'name\s*=\s*"([^"]+)"', content)
            for name in name_matches:
                if name and name not in ['', ' ']:
                    speaker_names.add(name)
            
            # Create mapping for each speaker in this file
            for speaker_name in speaker_names:
                key = f"{filename}_{speaker_name}"
                speaker_mapping[key] = {
                    'speaker_id': speaker_name,
                    'interview_number': filename,
                    'gender': "",  # Not available without TRS
                    'dialect': "",  # Not available without TRS
                    'is_multi_interview': False,  # Cannot determine without TRS
                }
                
        except Exception as e:
            print(f"   Warning: Could not parse TextGrid {filename}: {e}")
            continue
    
    return speaker_mapping


def get_speaker_info(speaker_mapping, filename, speaker_name):
    """
    Get consistent speaker information for a given filename and speaker.
    
    Args:
        speaker_mapping (dict): Speaker mapping from create_speaker_mapping()
        filename (str): Base filename (e.g., 'a018a')
        speaker_name (str): Speaker name from TextGrid
        
    Returns:
        dict: Speaker information with consistent IDs
    """
    key = f"{filename}_{speaker_name}"
    
    if key in speaker_mapping:
        return speaker_mapping[key]
    else:
        # Fallback for unmapped speakers
        return {
            'speaker_id': speaker_name,
            'interview_number': filename,
            'gender': "",
            'dialect': "",
            'is_multi_interview': False
        }


def print_speaker_statistics(speaker_mapping):
    """Print statistics about speaker mapping."""
    speakers_by_id = defaultdict(list)
    
    for key, info in speaker_mapping.items():
        speakers_by_id[info['speaker_id']].append(key)
    
    multi_interview = {sid: keys for sid, keys in speakers_by_id.items() if len(keys) > 1}
    
    print(f"\nSpeaker Statistics:")
    print(f"   Unique speaker IDs: {len(speakers_by_id)}")
    print(f"   Multi-interview speakers: {len(multi_interview)}")
    
    if multi_interview:
        print(f"\nMulti-interview speakers:")
        for speaker_id, appearances in list(multi_interview.items())[:5]:  # Show first 5
            interview_numbers = [speaker_mapping[key]['interview_number'] for key in appearances]
            print(f"   {speaker_id}: {len(appearances)} appearances in {sorted(set(interview_numbers))}")
        if len(multi_interview) > 5:
            print(f"   ... and {len(multi_interview) - 5} more")
    else:
        print(f"\nNo multi-interview speakers detected (this is normal when TRS files are not available)")
