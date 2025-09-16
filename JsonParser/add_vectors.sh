#!/bin/bash
# run_vectors_resume.sh
# Processes 0.json to 149.json, remembering where it left off.

PROGRESS_FILE="progress.txt"
START=0
END=149

# If progress file exists, resume from the next index
if [ -f "$PROGRESS_FILE" ]; then
    LAST_DONE=$(cat "$PROGRESS_FILE")
    START=$((LAST_DONE + 1))
    echo "Resuming from $START.json..."
else
    echo "Starting fresh..."
fi

for i in $(seq $START $END); do
    echo "Processing $i.json..."
    python3 vector_info.py "../Cheif_Delphi_JSONS/$i.json"

    # Save the current index after successful run
    if [ $? -eq 0 ]; then
        echo "$i" > "$PROGRESS_FILE"
    else
        echo "Error on $i.json, stopping."
        exit 1
    fi
done

echo "All files processed."
# Optional: remove progress file when finished
rm -f "$PROGRESS_FILE"
