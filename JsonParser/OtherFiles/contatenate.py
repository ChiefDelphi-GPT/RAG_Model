import os
import json
import shutil
import hashlib
from pathlib import Path

def get_json_hash(file_path):
    """Calculate hash of JSON content (normalized) to identify duplicates."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Convert to sorted JSON string for consistent hashing
        normalized = json.dumps(data, sort_keys=True)
        return hashlib.md5(normalized.encode()).hexdigest()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def find_duplicates_and_copy_unique(directories, output_dir):
    """
    Compare JSON files from multiple directories and copy unique files to output directory.
    
    Args:
        directories: List of source directories containing JSON files
        output_dir: Destination directory for unique JSON files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store hash -> (filename, file_path, directory_source)
    file_hashes = {}
    duplicate_hashes = set()
    
    # Track statistics per directory
    dir_stats = {}
    
    # Process all JSON files from all directories
    for dir_idx, directory in enumerate(directories, 1):
        if not os.path.exists(directory):
            print(f"Warning: Directory '{directory}' does not exist. Skipping.")
            dir_stats[directory] = {'exists': False, 'file_count': 0}
            continue
        
        # Get all JSON files in the directory
        try:
            json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        except PermissionError:
            print(f"Error: Permission denied for directory '{directory}'. Skipping.")
            dir_stats[directory] = {'exists': True, 'file_count': 0, 'error': 'Permission denied'}
            continue
        
        dir_stats[directory] = {'exists': True, 'file_count': len(json_files)}
        print(f"\nProcessing Directory {dir_idx}/{len(directories)} ({directory}): {len(json_files)} JSON files")
        
        for filename in json_files:
            file_path = os.path.join(directory, filename)
            file_hash = get_json_hash(file_path)
            
            if file_hash is None:
                continue
            
            if file_hash in file_hashes:
                # Duplicate found
                duplicate_hashes.add(file_hash)
                original = file_hashes[file_hash]
                print(f"  Duplicate: {filename} in {directory}")
                print(f"    -> matches {original[0]} in {original[2]}")
            else:
                # New unique file
                file_hashes[file_hash] = (filename, file_path, directory)
    
    # Copy unique files to output directory
    unique_count = 0
    print(f"\n{'='*70}")
    print("Copying unique files to output directory...")
    print(f"{'='*70}")
    
    for file_hash, (filename, file_path, source_dir) in file_hashes.items():
        if file_hash not in duplicate_hashes:
            dest_path = os.path.join(output_dir, filename)
            
            # Handle filename conflicts from different directories
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(output_dir, f"{base}_{counter}{ext}")
                    counter += 1
                print(f"Copied: {filename} -> {os.path.basename(dest_path)} (renamed due to conflict)")
            else:
                print(f"Copied: {filename}")
            
            shutil.copy2(file_path, dest_path)
            unique_count += 1
    
    # Print detailed summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"\nDirectory Statistics:")
    print(f"{'-'*70}")
    
    total_files = 0
    for directory, stats in dir_stats.items():
        if stats['exists']:
            file_count = stats['file_count']
            total_files += file_count
            print(f"  {directory}: {file_count} files")
            if 'error' in stats:
                print(f"    ⚠ {stats['error']}")
        else:
            print(f"  {directory}: Directory not found")
    
    print(f"\n{'-'*70}")
    print(f"Total files across all directories: {total_files}")
    print(f"Unique files (by content): {len(file_hashes)}")
    print(f"Duplicate files found: {len(duplicate_hashes)}")
    print(f"Unique files copied to output: {unique_count}")
    print(f"\nOutput directory: {output_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    DIRECTORIES = [
        "../../json_originals_0-149",
        "../../json_originals_150-1649",
        "../../json_originals_1650-4149",
        "../../json_originals_4150-6649",
        "../../json_originals_6650-9149",
        "../../json_originals_9150-11649",
        "../../json_originals_11650-14149",
        "../../json_originals_14150-16649",
    ]
    
    OUTPUT_DIR = "../../json_originals"
    
    print("JSON Duplicate Finder and Organizer")
    print("=" * 70)
    print(f"Scanning {len(DIRECTORIES)} directories for JSON files...")
    
    find_duplicates_and_copy_unique(DIRECTORIES, OUTPUT_DIR)