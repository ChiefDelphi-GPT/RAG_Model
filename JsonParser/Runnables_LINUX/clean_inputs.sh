#!/bin/bash
# run_clean_inputs_local.sh
# Process JSON files locally without any remote server.

set -e

PROGRESS_FILE="progress_clean_inputs.txt"
START=150
END=1151
STEP=3
INPUT_DIR="../../json_originals_150-1151"
OUTPUT_DIR="../../../RAG_Model/Cheif_Delphi_JSONS"

# Optional fresh run
if [ "$1" == "--fresh" ]; then
    echo "Starting a fresh run: deleting local progress file..."
    rm -f "$PROGRESS_FILE"
    echo "Fresh run setup complete."./
fi

cleanup_and_exit() {
    echo
    echo "Caught Ctrl+C! Updating local progress before exiting..."
    echo "Exiting..."
    exit 130
}

trap cleanup_and_exit SIGINT

# Determine start index
if [ ! -f "$PROGRESS_FILE" ]; then
    echo "$START" > "$PROGRESS_FILE"
    echo "Creating new progress file starting at $START.json..."
else
    LAST_DONE=$(cat "$PROGRESS_FILE")
    START=$((LAST_DONE))
    echo "Resuming from $START.json..."
fi

mkdir -p "$OUTPUT_DIR"

for i in $(seq $START $STEP $END); do
    echo "Processing group starting at $i..."

    # process i
    python3 ../input_cleaner.py "${INPUT_DIR}/${i}.json" \
        && echo "$i" > "$PROGRESS_FILE"

    # process i+1 if it exists
    if [ -f "${INPUT_DIR}/$((i + 1)).json" ]; then
        python3 ../input_cleaner.py "${INPUT_DIR}/$((i + 1)).json" \
            && echo "$((i + 1))" > "$PROGRESS_FILE"
    fi

    # process i+2 if it exists
    if [ -f "${INPUT_DIR}/$((i + 2)).json" ]; then
        python3 ../input_cleaner.py "${INPUT_DIR}/$((i + 2)).json" \
            && echo "$((i + 2))" > "$PROGRESS_FILE"
    fi
done

echo "All JSON files processed into $OUTPUT_DIR"
rm -f "$PROGRESS_FILE"
