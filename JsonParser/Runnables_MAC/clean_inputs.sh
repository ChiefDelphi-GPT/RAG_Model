#!/bin/bash
set -e

PROGRESS_FILE="progress_clean_inputs.txt"
START=150
END=1151
STEP=3

if [ "$1" == "--fresh" ]; then
    echo "Starting a fresh run: deleting local and remote progress files..."
    rm -f "$PROGRESS_FILE"
    ssh fe.ds "rm -f /home/rhayrapetyan/automatic/progress_clean_inputs.txt"
    echo "Fresh run setup complete."
fi

retry_scp () {
    local src=$1
    local dest=$2
    local delay=5
    while true; do
        if scp -r "$src" "$dest"; then
            echo "Successfully copied $src to $dest"
            break
        else
            echo "Copy failed for $src → $dest, retrying in ${delay}s..."
            sleep $delay
        fi
    done
}

cleanup_and_exit() {
    echo ""
    echo "Caught Ctrl+C! Updating local progress before exiting..."

    # Keep trying to copy the remote progress file
    retry_scp "fe.ds:/home/rhayrapetyan/automatic/progress_clean_inputs.txt" "$PROGRESS_FILE"

    # Keep trying to copy all processed JSON files
    retry_scp "fe.ds:/home/rhayrapetyan/automatic/Cheif_Delphi_JSONS/*" \
              "../../../RAG_Model/Cheif_Delphi_JSONS/"

    echo "Exiting..."
    exit 130
}

# Set up trap to catch Ctrl+C (SIGINT)
trap cleanup_and_exit SIGINT

if [ ! -f "$PROGRESS_FILE" ]; then
    echo "$START" > "$PROGRESS_FILE"
    echo "Creating new progress file starting at $START.json..."
fi

if [ -f "$PROGRESS_FILE" ]; then
    scp "$PROGRESS_FILE" fe.ds:/home/rhayrapetyan/automatic/
fi

if [ -f "$PROGRESS_FILE" ]; then
    LAST_DONE=$(cat "$PROGRESS_FILE")
    START=$((LAST_DONE))
    echo "Resuming from $START.json..."
else
    echo "Starting fresh..."
fi

# Copy data directory to remote source
./delete_remote.sh
scp -r ../../json_originals_150-1151 fe.ds:/home/rhayrapetyan/automatic/
ssh fe.ds "mkdir -p /home/rhayrapetyan/automatic/JsonParser"
scp ../input_cleaner.py fe.ds:/home/rhayrapetyan/automatic/JsonParser/
scp $PROGRESS_FILE fe.ds:/home/rhayrapetyan/automatic/ || true

# SSH and execute commands on remote
ssh fe.ds << 'EOF'
set -e

PROGRESS_FILE="/home/rhayrapetyan/automatic/progress_clean_inputs.txt"

if [ -f "$PROGRESS_FILE" ]; then
    LAST_DONE=$(cat "$PROGRESS_FILE")
    START=$((LAST_DONE))
    echo "Resuming from $START.json..."
else
    START=150
    echo "Starting fresh..."
fi

END=1151
STEP=3

cd /home/rhayrapetyan/automatic/
mkdir -p /home/rhayrapetyan/automatic/Cheif_Delphi_JSONS

# Create the run.sh script with proper escaping
cat > run.sh << 'EORUN'
#!/bin/bash
cd JsonParser/

START=150
END=1151
STEP=3

PROGRESS_FILE="/home/rhayrapetyan/automatic/progress_clean_inputs.txt"
if [ -f "$PROGRESS_FILE" ]; then
    LAST_DONE=$(cat "$PROGRESS_FILE")
    START=$((LAST_DONE))
fi

for i in $(seq $START $STEP $END); do
    srun -p general --mem=800G --ntasks=1 -t 6:00:00 --gres=gpu:1 \
        bash -c "
            python3 input_cleaner.py ../json_originals_150-1151/${i}.json && echo ${i} > $PROGRESS_FILE

            if [ -f ../json_originals_150-1151/\$((${i} + 1)).json ]; then
                python3 input_cleaner.py ../json_originals_150-1151/\$((${i} + 1)).json && echo \$((${i} + 1)) > $PROGRESS_FILE
            fi

            if [ -f ../json_originals_150-1151/\$((${i} + 2)).json ]; then
                python3 input_cleaner.py ../json_originals_150-1151/\$((${i} + 2)).json && echo \$((${i} + 2)) > $PROGRESS_FILE
            fi
        "
done
EORUN

chmod +x run.sh

# Run the processing script
./run.sh

exit
EOF


# Copy output directory back to local
scp -r 'fe.ds:/home/rhayrapetyan/automatic/Cheif_Delphi_JSONS/*' ../../../RAG_Model/Cheif_Delphi_JSONS/

echo "All JSON files processed and copied to Cheif_Delphi_JSONS directory."
rm -f "$PROGRESS_FILE"