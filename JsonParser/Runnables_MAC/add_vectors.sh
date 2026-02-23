#!/bin/bash
# run_vectors_resume_remote_cluster.sh
# Runs locally but executes everything on fe.ds cluster in background

REMOTE_HOST="fe.ds"
REMOTE_BASE_DIR="/home/rhayrapetyan/automatic"
REMOTE_SCRIPT_DIR="$REMOTE_BASE_DIR/scripts"
REMOTE_JSON_DIR="$REMOTE_BASE_DIR/Cheif_Delphi_JSONS"
REMOTE_LOGS_DIR="$REMOTE_BASE_DIR/logs"
REMOTE_PROGRESS_FILE="$REMOTE_BASE_DIR/progress_adding_vectors.txt"
REMOTE_MAIN_LOG="$REMOTE_BASE_DIR/main_processing.log"
REMOTE_PID_FILE="$REMOTE_BASE_DIR/processing.pid"
LOCAL_PROGRESS_FILE="./progress_adding_vectors.txt"

START=0
END=166490

echo "Setting up directory structure on $REMOTE_HOST..."

# Create directory structure on remote cluster
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_SCRIPT_DIR $REMOTE_JSON_DIR $REMOTE_LOGS_DIR"

if [ $? -ne 0 ]; then
    echo -e "\033[0;31mFailed to create directories on $REMOTE_HOST\033[0m"
    exit 1
fi

echo "Directory structure created successfully."

# Copy the vector_info.py script to the remote cluster if it exists locally
if [ -f "../vector_info.py" ]; then
    echo "Copying vector_info.py to remote cluster..."
    scp ../vector_info.py "$REMOTE_HOST:$REMOTE_SCRIPT_DIR/"
fi

# Copy all JSON files to remote cluster (do this once at the beginning)
echo "Copying JSON files to remote cluster..."
if [ -d "../../Cheif_Delphi_JSONS" ]; then
    scp -r ../../Cheif_Delphi_JSONS/* "$REMOTE_HOST:$REMOTE_JSON_DIR/"
    if [ $? -ne 0 ]; then
        echo -e "\033[0;31mFailed to copy JSON files to remote cluster\033[0m"
        exit 1
    fi
    echo "JSON files copied successfully."
fi
``
# Create the processing script that will run on the remote cluster
cat > /tmp/remote_process.sh << 'EOFSCRIPT'
#!/bin/bash

REMOTE_BASE_DIR="$HOME/FRC_Processing"
REMOTE_SCRIPT_DIR="$REMOTE_BASE_DIR/scripts"
REMOTE_JSON_DIR="$REMOTE_BASE_DIR/Cheif_Delphi_JSONS"
REMOTE_LOGS_DIR="$REMOTE_BASE_DIR/logs"
REMOTE_PROGRESS_FILE="$REMOTE_BASE_DIR/progress_adding_vectors.txt"

START=0
END=166490

# If progress file exists, resume from the next index
if [ -f "$REMOTE_PROGRESS_FILE" ]; then
    LAST_DONE=$(cat "$REMOTE_PROGRESS_FILE")
    START=$((LAST_DONE + 1))
    echo "Resuming from $START.json..." >&2
else
    echo "Starting fresh..." >&2
fi

for ((i=START; i<=END; i++)); do
    echo "[$(date)] Processing $i.json..." >&2
    
    # Run the Python script using srun
    srun -p general --mem=500G --ntasks=8 -t 12:00:00 --gres=gpu:1 \
        python3 "$REMOTE_SCRIPT_DIR/vector_info.py" "$REMOTE_JSON_DIR/$i.json" \
        > "$REMOTE_LOGS_DIR/output_$i.log" 2> "$REMOTE_LOGS_DIR/error_$i.log"
    
    EXIT_CODE=$?
    
    # Check if the command was successful
    if [ $EXIT_CODE -eq 0 ]; then
        # Save the current index after successful run
        echo "$i" > "$REMOTE_PROGRESS_FILE"
        echo "[$(date)] $i.json finished successfully." >&2
    else
        echo "[$(date)] Error on $i.json (exit code: $EXIT_CODE), stopping." >&2
        exit 1
    fi
done

echo "[$(date)] All files processed successfully!" >&2

# Remove progress file when finished
if [ -f "$REMOTE_PROGRESS_FILE" ]; then
    rm "$REMOTE_PROGRESS_FILE"
fi
EOFSCRIPT

# Copy the processing script to remote cluster
scp /tmp/remote_process.sh "$REMOTE_HOST:$REMOTE_SCRIPT_DIR/"
rm /tmp/remote_process.sh

# Make it executable on remote
ssh "$REMOTE_HOST" "chmod +x $REMOTE_SCRIPT_DIR/remote_process.sh"

echo "Starting background processing on $REMOTE_HOST..."

# Start the processing script in a nohup session on remote cluster
ssh "$REMOTE_HOST" "cd $REMOTE_SCRIPT_DIR && nohup ./remote_process.sh > $REMOTE_MAIN_LOG 2>&1 & echo \$! > $REMOTE_PID_FILE"

if [ $? -eq 0 ]; then
    echo -e "\033[0;32mProcessing started successfully on $REMOTE_HOST\033[0m"
    echo ""
    echo "===== IMPORTANT INFORMATION ====="
    echo "Main log file: $REMOTE_HOST:$REMOTE_MAIN_LOG"
    echo "Progress file: $REMOTE_HOST:$REMOTE_PROGRESS_FILE"
    echo "Individual logs: $REMOTE_HOST:$REMOTE_LOGS_DIR/"
    echo "PID file: $REMOTE_HOST:$REMOTE_PID_FILE"
    echo ""
    echo "To check progress:"
    echo "  ssh $REMOTE_HOST 'tail -f $REMOTE_MAIN_LOG'"
    echo "  ssh $REMOTE_HOST 'cat $REMOTE_PROGRESS_FILE'"
    echo ""
    echo "To stop the processing:"
    echo "  ssh $REMOTE_HOST 'kill \$(cat $REMOTE_PID_FILE) && rm $REMOTE_PID_FILE'"
    echo ""
    echo "You can now safely disconnect. The process will continue running."
    echo "================================="
else
    echo -e "\033[0;31mFailed to start processing on $REMOTE_HOST\033[0m"
    exit 1
fi