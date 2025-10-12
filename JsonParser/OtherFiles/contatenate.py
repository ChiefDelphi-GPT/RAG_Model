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

def find_duplicates_and_copy_unique(dir1, dir2, dir3, output_dir):
    """
    Compare JSON files from 3 directories and copy unique files to output directory.
    
    Args:
        dir1, dir2, dir3: Source directories containing JSON files
        output_dir: Destination directory for unique JSON files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store hash -> (file_path, directory_source)
    file_hashes = {}
    duplicate_hashes = set()
    
    directories = [dir1, dir2, dir3]
    
    # Process all JSON files from all directories
    for dir_idx, directory in enumerate(directories, 1):
        if not os.path.exists(directory):
            print(f"Warning: Directory '{directory}' does not exist. Skipping.")
            continue
            
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        print(f"\nProcessing Directory {dir_idx} ({directory}): {len(json_files)} JSON files")
        
        for filename in json_files:
            file_path = os.path.join(directory, filename)
            file_hash = get_json_hash(file_path)
            
            if file_hash is None:
                continue
            
            if file_hash in file_hashes:
                # Duplicate found
                duplicate_hashes.add(file_hash)
                original = file_hashes[file_hash]
                print(f"  Duplicate found: {filename} (matches {original[0]})")
            else:
                # New unique file
                file_hashes[file_hash] = (filename, file_path, directory)
    
    # Copy unique files to output directory
    unique_count = 0
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
            
            shutil.copy2(file_path, dest_path)
            unique_count += 1
            print(f"Copied: {filename} -> {output_dir}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {len(file_hashes)}")
    print(f"Duplicate files found: {len(duplicate_hashes)}")
    print(f"Unique files copied: {unique_count}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    # Define your directories here
    DIR1 = "directory1"
    DIR2 = "directory2"
    DIR3 = "directory3"
    OUTPUT_DIR = "unique_jsons"
    
    print("JSON Duplicate Finder and Organizer")
    print("=" * 60)
    
    find_duplicates_and_copy_unique(DIR1, DIR2, DIR3, OUTPUT_DIR)