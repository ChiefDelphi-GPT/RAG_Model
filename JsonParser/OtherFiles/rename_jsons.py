import os
import sys

def renumber_json_files(target_dir, start=1, end=None):
    """
    Renumber JSON files sequentially from START to END.
    Example: If you have files 100.json, 101.json, 102.json
             and start=1, end=3, they become 1.json, 2.json, 3.json
    
    Args:
        target_dir: Directory containing JSON files
        start: Starting number (default: 1)
        end: Ending number (if None, will be start + num_files - 1)
    """
    # Collect numeric-json files
    files = []
    for f in os.listdir(target_dir):
        name, ext = os.path.splitext(f)
        if ext.lower() == ".json" and name.isdigit():
            files.append((int(name), f))

    if not files:
        print("No numeric JSON files found.")
        return

    # Sort by original number
    files.sort(key=lambda x: x[0])
    
    # Determine end if not specified
    if end is None:
        end = start + len(files) - 1
    
    # Check if we have the right number of files
    expected_count = end - start + 1
    if len(files) != expected_count:
        print(f"Warning: Found {len(files)} files but expected {expected_count} files")
        print(f"(START={start}, END={end} means {expected_count} files)")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # First pass: rename to temporary names to avoid conflicts
    temp_renames = []
    for idx, (old_num, fname) in enumerate(files):
        new_num = start + idx
        temp_name = f"temp_{new_num}_{old_num}.json"
        src = os.path.join(target_dir, fname)
        temp_dst = os.path.join(target_dir, temp_name)
        
        os.rename(src, temp_dst)
        temp_renames.append((temp_name, new_num, fname))
        print(f"Temp rename: {fname} -> {temp_name}")
    
    # Second pass: rename from temp names to final names
    print("\nFinal renaming:")
    for temp_name, new_num, original_name in temp_renames:
        new_name = f"{new_num}.json"
        temp_src = os.path.join(target_dir, temp_name)
        dst = os.path.join(target_dir, new_name)
        
        if os.path.exists(dst) and dst != temp_src:
            print(f"ERROR: {new_name} already exists. Aborting.")
            # Rollback temp files
            print("Rolling back changes...")
            for t_name, _, orig_name in temp_renames:
                t_path = os.path.join(target_dir, t_name)
                if os.path.exists(t_path):
                    os.rename(t_path, os.path.join(target_dir, orig_name))
            return
        
        os.rename(temp_src, dst)
        print(f"{original_name} -> {new_name}")
    
    print(f"\nSuccessfully renumbered {len(files)} files from {start} to {end}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rename_jsons.py /path/to/json/folder [START] [END]")
        print("Example: python rename_jsons.py ./data 1 100")
        print("         (renumbers all JSON files to 1.json through 100.json)")
    else:
        target_dir = sys.argv[1]
        start = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        end = int(sys.argv[3]) if len(sys.argv) > 3 else None
        
        renumber_json_files(target_dir, start, end)