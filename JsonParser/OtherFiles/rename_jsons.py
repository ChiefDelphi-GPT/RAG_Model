import os
import sys

def rename_json_files(target_dir):
    """
    Add 149 to the numeric part of each .json filename in target_dir.
    Example: 1.json -> 150.json, 1500.json -> 1649.json
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

    # Sort descending to avoid overwriting (e.g. 10 -> 159 before 1 -> 150)
    files.sort(key=lambda x: x[0], reverse=True)

    for num, fname in files:
        new_num = num + 149
        new_name = f"{new_num}.json"
        src = os.path.join(target_dir, fname)
        dst = os.path.join(target_dir, new_name)

        if os.path.exists(dst):
            print(f"Skipping {fname}: {new_name} already exists.")
            continue

        os.rename(src, dst)
        print(f"{fname} -> {new_name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rename_jsons.py /path/to/json/folder")
    else:
        rename_json_files(sys.argv[1])
