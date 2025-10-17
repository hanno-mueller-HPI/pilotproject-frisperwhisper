#!/usr/bin/env python3
"""
Transform ESLO TextGrid files to LangAge format.

This script processes all ESLO TextGrid files in the data/LangAgeESLOcombined16kHz folder,
applies the same transformations as eslo2langAge.py, and overwrites the original files
with the transformed content.

ESLO files are identified by filenames starting with "ESLO1_ENT_" or "ESLO2_ENT_".
"""

import os
import re
from pathlib import Path


def identify_eslo_textgrids(input_folder):
    """
    Find all ESLO TextGrid files in the specified folder.
    
    Args:
        input_folder (Path): Path to the folder containing TextGrid files
        
    Returns:
        list: List of Path objects for ESLO TextGrid files
    """
    eslo_files = []
    
    for file_path in input_folder.iterdir():
        if (file_path.is_file() and 
            file_path.name.endswith('.TextGrid') and 
            (file_path.name.startswith('ESLO1_ENT_') or file_path.name.startswith('ESLO2_ENT_'))):
            eslo_files.append(file_path)
    
    return sorted(eslo_files)


def apply_eslo_to_langage_transformations(content):
    """
    Apply all the text transformations from eslo2langAge.py to convert ESLO format
    to LangAge format.
    
    Args:
        content (str): Original TextGrid file content
        
    Returns:
        str: Transformed content
    """
    modified_content = content
    
    # Apply all regex replacements from the original script
    modified_content = re.sub(r'\(rire\)', '(rir)', modified_content)            
    modified_content = re.sub(r'\&amp;', '', modified_content)            
    modified_content = re.sub(r'\(rire en fond\)', '(rir)', modified_content)            
    modified_content = re.sub(r'\(sonnerie\)', '(bru)', modified_content)            
    modified_content = re.sub(r'\(alumettes\)', '(bru)', modified_content)            
    modified_content = re.sub(r'\(pi\)', 'XXX', modified_content)            
    modified_content = re.sub(r'\(pif\)', 'XXX', modified_content)            
    modified_content = re.sub(r'\(euh\)', 'euh', modified_content)            
    modified_content = re.sub(r'\(mah\)', 'mah', modified_content)
    modified_content = re.sub(r'\(aec\)', '', modified_content)
    modified_content = re.sub(r'\(afin\)', '', modified_content)
    modified_content = re.sub(r'\(amobinable\)', '', modified_content)
    modified_content = re.sub(r'\(applelon\)', '', modified_content)
    modified_content = re.sub(r'\(ar\)', '', modified_content)
    modified_content = re.sub(r'\(architecturellement\)', '', modified_content)
    modified_content = re.sub(r'\(arriverez\)', '', modified_content)
    modified_content = re.sub(r'\(autoute\)', '', modified_content)
    modified_content = re.sub(r'\(aye\)', '', modified_content)
    modified_content = re.sub(r'\(azociation\)', '', modified_content)
    modified_content = re.sub(r'\(babidtonne\)', '', modified_content)
    modified_content = re.sub(r'\(BafD\)', '', modified_content)
    modified_content = re.sub(r'\(batterie\)', '', modified_content)
    modified_content = re.sub(r'\(beaulieu\)', '', modified_content)
    modified_content = re.sub(r'\(bg\)', '', modified_content)
    modified_content = re.sub(r'\(boah\)', '', modified_content)
    modified_content = re.sub(r'\(boit\)', '', modified_content)
    modified_content = re.sub(r'\(brides\)', '', modified_content)
    modified_content = re.sub(r'\(campis\)', '', modified_content)
    modified_content = re.sub(r'\(ch\)', '', modified_content)
    modified_content = re.sub(r'\(chais\)', '', modified_content)
    modified_content = re.sub(r'\(chartre\)', '', modified_content)
    modified_content = re.sub(r'\(chuis\)', '', modified_content)
    modified_content = re.sub(r'\(circulation\)', '', modified_content)
    modified_content = re.sub(r'\(concous\)', '', modified_content)
    modified_content = re.sub(r'\(conv\)', '', modified_content)
    modified_content = re.sub(r'\(cor\)', '', modified_content)
    modified_content = re.sub(r'\(coupaib\)', '', modified_content)
    modified_content = re.sub(r'\(couser\)', '', modified_content)
    modified_content = re.sub(r'\(couta\)', '', modified_content)
    modified_content = re.sub(r'\(cri\)', '', modified_content)
    modified_content = re.sub(r'\(cris\)', '', modified_content)
    modified_content = re.sub(r'\(croyent\)', '', modified_content)
    modified_content = re.sub(r'\(cte\)', '', modified_content)
    modified_content = re.sub(r'\(dans\)', '', modified_content)
    modified_content = re.sub(r'\(designer\)', '', modified_content)
    modified_content = re.sub(r'\(djeuns\)', '', modified_content)
    modified_content = re.sub(r'\(doit\)', '', modified_content)
    modified_content = re.sub(r'\(drans\)', '', modified_content)
    modified_content = re.sub(r'\(e\)', '', modified_content)
    modified_content = re.sub(r'\(eccetera\)', '', modified_content)
    modified_content = re.sub(r'\(en\)', '', modified_content)
    modified_content = re.sub(r'\(enpcore\)', '', modified_content)
    modified_content = re.sub(r'\(essayent\)', '', modified_content)
    modified_content = re.sub(r'\(extenction\)', '', modified_content)
    modified_content = re.sub(r'\(fas\)', '', modified_content)
    modified_content = re.sub(r'\(fin\)', '', modified_content)
    modified_content = re.sub(r'\(fon\)', '', modified_content)
    modified_content = re.sub(r'\(frits\)', '', modified_content)
    modified_content = re.sub(r'\(jingle\)', '', modified_content)
    modified_content = re.sub(r'\(klaxon\)', '', modified_content)
    modified_content = re.sub(r'\(le\)', '', modified_content)
    modified_content = re.sub(r'\(les\)', '', modified_content)
    modified_content = re.sub(r'\(maker\)', '', modified_content)
    modified_content = re.sub(r'\(manifations\)', '', modified_content)
    modified_content = re.sub(r'\(maritainement\)', '', modified_content)
    modified_content = re.sub(r'\(masterz\)', '', modified_content)
    modified_content = re.sub(r'\(menfin\)', '', modified_content)
    modified_content = re.sub(r'\(morne\)', '', modified_content)
    modified_content = re.sub(r'\(mouais\)', '', modified_content)
    modified_content = re.sub(r'\(moui\)', '', modified_content)
    modified_content = re.sub(r'\(muis\)', '', modified_content)
    modified_content = re.sub(r'\(n\)', '', modified_content)
    modified_content = re.sub(r'\(ne\)', '', modified_content)
    modified_content = re.sub(r'\(nontrans\)', '', modified_content)
    modified_content = re.sub(r'\(Nyoung\)', '', modified_content)
    modified_content = re.sub(r'\(oeuf\)', '', modified_content)
    modified_content = re.sub(r'\(on\)', '', modified_content)
    modified_content = re.sub(r'\(pa\)', '', modified_content)
    modified_content = re.sub(r'\(pala\)', '', modified_content)
    modified_content = re.sub(r'\(pap\)', '', modified_content)
    modified_content = re.sub(r'\(pas\)', '', modified_content)
    modified_content = re.sub(r'\(pf\)', '', modified_content)
    modified_content = re.sub(r'\(pg\)', '', modified_content)
    modified_content = re.sub(r'\(pis\)', '', modified_content)
    modified_content = re.sub(r'\(pneumeul\)', '', modified_content)
    modified_content = re.sub(r'\(po\)', '', modified_content)
    modified_content = re.sub(r'\(popilaires\)', '', modified_content)
    modified_content = re.sub(r'\(promenez\)', '', modified_content)
    modified_content = re.sub(r'\(quarter\)', '', modified_content)
    modified_content = re.sub(r'\(quates\)', '', modified_content)
    modified_content = re.sub(r'\(que\)', '', modified_content)
    modified_content = re.sub(r'\(quouquoi\)', '', modified_content)
    modified_content = re.sub(r'\(ranto\)', '', modified_content)
    modified_content = re.sub(r'\(reniflement\)', '', modified_content)
    modified_content = re.sub(r'\(reprosse\)', '', modified_content)
    modified_content = re.sub(r'\(revoyent\)', '', modified_content)
    modified_content = re.sub(r'\(rigougreux\)', '', modified_content)
    modified_content = re.sub(r'\(roh\)', '', modified_content)
    modified_content = re.sub(r'\(sasse\)', '', modified_content)
    modified_content = re.sub(r'\(sasso\)', '', modified_content)
    modified_content = re.sub(r'\(serais\)', '', modified_content)
    modified_content = re.sub(r'\(sien\)', '', modified_content)
    modified_content = re.sub(r'\(sif\)', '', modified_content)
    modified_content = re.sub(r'\(sixe\)', '', modified_content)
    modified_content = re.sub(r'\(surout\)', '', modified_content)
    modified_content = re.sub(r'\(terer\)', '', modified_content)
    modified_content = re.sub(r'\(tous\)', '', modified_content)
    modified_content = re.sub(r'\(train\)', '', modified_content)
    modified_content = re.sub(r'\(tram\)', '', modified_content)
    modified_content = re.sub(r'\(troufe\)', '', modified_content)
    modified_content = re.sub(r'\(tx\)', '', modified_content)
    modified_content = re.sub(r'\(ui\)', '', modified_content)
    modified_content = re.sub(r'\(un\)', '', modified_content)
    modified_content = re.sub(r'\(vent\)', '', modified_content)
    modified_content = re.sub(r'\(ver\)', '', modified_content)
    modified_content = re.sub(r'\(veut\)', '', modified_content)
    modified_content = re.sub(r'\(vingts\)', '', modified_content)
    modified_content = re.sub(r'\(visider\)', '', modified_content)
    modified_content = re.sub(r'\(voix\)', '', modified_content)
    modified_content = re.sub(r'\(vouais\)', '', modified_content)
    modified_content = re.sub(r'\(voyger\)', '', modified_content)
    modified_content = re.sub(r'\(wingles\)', '', modified_content)
    modified_content = re.sub(r'\(yes\)', '', modified_content)
    modified_content = re.sub(r'\(zont\)', '', modified_content)
    modified_content = re.sub(r'\(zy\)', '', modified_content)
    modified_content = re.sub(r'<Who nb="\d+"\s*/>', '', modified_content)
    modified_content = re.sub(r'Ouille', 'ouille', modified_content)
    modified_content = re.sub(r'- ', '', modified_content)           
    
    # Add space after apostrophe, except in specific expressions
    # Use negative lookbehind to exclude specific words
    modified_content = re.sub(
        r"(?<!aujourd)(?<!d)(?<!c)(?<!n)(?<!quelqu)'(?! )",
        "' ",
        modified_content
    )
    
    # But protect the specific expressions by temporarily marking them
    protected_words = [
        "aujourd'hui", "c'est-à-dire", "d'abord", "d'accord", 
        "d'ailleurs", "d'autant", "d'habitude", "d'œuvre", 
        "n'est-ce-pas", "quelqu'un"
    ]
    
    # Replace apostrophes in protected words back (remove added space)
    for word in protected_words:
        modified_content = re.sub(word.replace("'", "' "), word, modified_content)
    
    return modified_content


def transform_eslo_textgrids():
    """
    Main function to process all ESLO TextGrid files in the target folder.
    """
    # Define the input folder path
    input_folder = Path("/sc/home/hanno.mueller/pilotproject-frisperwhisper/data/LangAgeESLOcombined16kHz")
    
    if not input_folder.exists():
        print(f"Error: Input folder does not exist: {input_folder}")
        return
    
    # Find all ESLO TextGrid files
    eslo_files = identify_eslo_textgrids(input_folder)
    
    if not eslo_files:
        print("No ESLO TextGrid files found in the input folder.")
        return
    
    print(f"Found {len(eslo_files)} ESLO TextGrid file(s) to process:")
    for file_path in eslo_files:
        print(f"  - {file_path.name}")
    print()
    
    # Process each file
    processed_count = 0
    error_count = 0
    
    for file_path in eslo_files:
        try:
            # Read the original file
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Apply transformations
            transformed_content = apply_eslo_to_langage_transformations(original_content)
            
            # Overwrite the original file with transformed content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(transformed_content)
            
            print(f"✓ Processed: {file_path.name}")
            processed_count += 1
            
        except Exception as e:
            print(f"✗ Error processing {file_path.name}: {str(e)}")
            error_count += 1
    
    print(f"\nTransformation complete!")
    print(f"Successfully processed: {processed_count} files")
    if error_count > 0:
        print(f"Errors encountered: {error_count} files")
    
    print(f"\nAll ESLO TextGrid files in {input_folder} have been transformed to LangAge format.")


if __name__ == "__main__":
    transform_eslo_textgrids()