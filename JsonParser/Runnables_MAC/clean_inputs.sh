#!/bin/bash
# run_vectors_resume_local_single.sh
# Processes JSON files locally with progress tracking (one at a time)

set -e

# -------------------
# Configuration
# -------------------
SCRIPT_DIR="/Users/rubenhayrapetyan/Downloads/Code/FRC/CheifDelphi-GPT/RAG_Model/JsonParser/Runnables_MAC"
PROGRESS_FILE="$SCRIPT_DIR/../../progress_clean_inputs.txt"
START=1650
END=16649

# Directories relative to this script
INPUT_DIR="$SCRIPT_DIR/../../json_originals"
OUTPUT_DIR="$SCRIPT_DIR/../../../RAG_Model/Cheif_Delphi_JSONS"
CLEANER="$SCRIPT_DIR/../input_cleaner.py"

# -------------------
# Handle --fresh run
# -------------------
if [ "$1" == "--fresh" ]; then
    echo "Starting a fresh run: deleting local progress file..."
    rm -f "$PROGRESS_FILE"
    echo "Fresh run setup complete."
fi

# -------------------
# Cleanup on Ctrl+C
# -------------------
CURRENT_INDEX=$START

cleanup() {
    echo ""
    echo "Caught Ctrl+C! Saving progress at index $CURRENT_INDEX"
    echo "$CURRENT_INDEX" > "$PROGRESS_FILE"
    exit 130
}

trap cleanup INT TERM

# -------------------
# Initialize / resume
# -------------------
if [ ! -f "$PROGRESS_FILE" ]; then
    echo "$START" > "$PROGRESS_FILE"
    echo "Creating new progress file starting at $START.json..."
else
    LAST_DONE=$(cat "$PROGRESS_FILE" | tr -d '[:space:]')
    if [ -n "$LAST_DONE" ]; then
        START=$((LAST_DONE))
        echo "Resuming from $START.json..."
    fi
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# -------------------
# Main processing loop
# -------------------
for i in $(seq $START $END); do
    CURRENT_INDEX=$i
    INPUT_FILE="$INPUT_DIR/$i.json"

    if [ -f "$INPUT_FILE" ]; then
        echo "Processing $i.json..."
        python3 "$CLEANER" "$INPUT_FILE"
        echo "$i" > "$PROGRESS_FILE"
    else
        echo "$i.json does not exist, skipping..."
    fi
done

echo "All JSON files processed into $OUTPUT_DIR"
rm -f "$PROGRESS_FILE"