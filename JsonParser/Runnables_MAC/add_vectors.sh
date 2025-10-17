#!/bin/bash
# add_vectors.sh
# Processes 0.json to 16649.json, remembering where it left off.

set -e

START=0
END=16649
LOCAL_DIR="../../Cheif_Delphi_JSONS"
REMOTE_DIR="/home/rhayrapetyan/automatic/Cheif_Delphi_JSONS"
PROGRESS_FILE="progress_add_vectors.txt"  # Relative to REMOTE_DIR

# Create directories if they don't exist
ssh fe.ds "mkdir -p /home/rhayrapetyan/automatic"
ssh fe.ds "mkdir -p $REMOTE_DIR"

# Check if files already exist on remote
echo "Checking if files already exist on remote server..."
FILES_EXIST=$(ssh fe.ds "[ -d '$REMOTE_DIR' ] && [ \$(find '$REMOTE_DIR' -name '*.json' -type f | wc -l) -gt 0 ] && echo 'yes' || echo 'no'")

if [ "$FILES_EXIST" = "yes" ]; then
    echo "Files already exist on remote server. Skipping transfer."
    REMOTE_FILE_COUNT=$(ssh fe.ds "find '$REMOTE_DIR' -name '*.json' -type f | wc -l")
    echo "Found $REMOTE_FILE_COUNT JSON files on remote server."
else
    echo "No files found on remote. Transferring files from $START.json to $END.json..."
    cd "$LOCAL_DIR"

    # Create list of files that exist
    files=()
    for i in $(seq $START $END); do
        if [ -f "$i.json" ]; then
            files+=("$i.json")
        fi
    done

    if [ ${#files[@]} -gt 0 ]; then
        echo "Found ${#files[@]} files to transfer..."
        tar --no-xattrs --no-mac-metadata --exclude='*.txt' -czf - "${files[@]}" | \
            ssh fe.ds "cd $REMOTE_DIR && tar xzf -"
        echo "Transfer complete."
    else
        echo "No files found in range."
        exit 1
    fi
fi

# Always ensure vector_info.py is up to date
ssh fe.ds "mkdir -p /home/rhayrapetyan/automatic/JsonParser"
scp /Users/rubenhayrapetyan/Downloads/Code/FRC/CheifDelphi-GPT/RAG_Model/JsonParser/vector_info.py fe.ds:/home/rhayrapetyan/automatic/JsonParser/

ssh fe.ds << 'EOF'
set -e
cd /home/rhayrapetyan/automatic/Cheif_Delphi_JSONS

START=0
END=16649
PROGRESS_FILE="progress_add_vectors.txt"

# Function to safely read progress
read_progress() {
    if [ -f "$PROGRESS_FILE" ] && [ -s "$PROGRESS_FILE" ]; then
        cat "$PROGRESS_FILE"
    else
        echo "-1"
    fi
}

# Function to safely write progress (atomic)
write_progress() {
    local value=$1
    echo "$value" > "${PROGRESS_FILE}.tmp"
    mv "${PROGRESS_FILE}.tmp" "$PROGRESS_FILE"
}

# Initialize progress file if needed
if [ ! -f "$PROGRESS_FILE" ] || [ ! -s "$PROGRESS_FILE" ]; then
    write_progress "-1"
    echo "Initialized progress file at -1"
else
    CURRENT_PROGRESS=$(read_progress)
    echo "Progress file found. Resuming from file $CURRENT_PROGRESS (next: $((CURRENT_PROGRESS + 1)))"
fi

# Main processing loop
while true; do
    LAST_DONE=$(read_progress)
    CURRENT=$((LAST_DONE + 1))
    
    # Check if we're done
    if [ "$CURRENT" -gt "$END" ]; then
        echo "All files from $START to $END have been processed!"
        break
    fi
    
    echo "=== Starting new 12-hour batch from file $CURRENT ==="
    
    # Submit job and wait for it to complete
    srun -p general --mem=400G -n 1 -t 12:00:00 --gres=gpu:1 bash -c "
        cd /home/rhayrapetyan/automatic/Cheif_Delphi_JSONS
        
        PROGRESS_FILE='progress_add_vectors.txt'
        
        # Function to safely read progress inside job
        read_progress() {
            if [ -f \"\$PROGRESS_FILE\" ] && [ -s \"\$PROGRESS_FILE\" ]; then
                cat \"\$PROGRESS_FILE\"
            else
                echo \"-1\"
            fi
        }
        
        # Function to safely write progress inside job
        write_progress() {
            local value=\$1
            echo \"\$value\" > \"\${PROGRESS_FILE}.tmp\"
            mv \"\${PROGRESS_FILE}.tmp\" \"\$PROGRESS_FILE\"
        }
        
        # Read starting point inside the job
        LAST_DONE=\$(read_progress)
        START_FROM=\$((LAST_DONE + 1))
        
        echo \"Job starting from file \$START_FROM\"
        
        for i in \$(seq \$START_FROM $END); do
            JSON_FILE=\"\$i.json\"
            
            # Check if file exists
            if [ ! -f \"\$JSON_FILE\" ]; then
                echo \"Warning: \$JSON_FILE not found, skipping (not updating progress)...\"
                continue
            fi
            
            echo \"Processing \$JSON_FILE...\"
            
            # Try to process the file
            if python3 ../JsonParser/vector_info.py \"\$JSON_FILE\"; then
                # Only update progress if processing succeeded
                write_progress \"\$i\"
                echo \"Successfully processed \$JSON_FILE\"
            else
                echo \"Error processing \$JSON_FILE - stopping to prevent data loss\"
                exit 1
            fi
        done
        
        FINAL_PROGRESS=\$(read_progress)
        echo \"Batch complete. Last file processed: \$FINAL_PROGRESS\"
    "
    
    # Check job exit status
    JOB_STATUS=$?
    if [ $JOB_STATUS -ne 0 ]; then
        echo "Job failed with status $JOB_STATUS. Stopping."
        exit 1
    fi
    
    # After job completes, check progress and loop
    echo "=== 12-hour batch completed ==="
    LAST_DONE=$(read_progress)
    echo "Last completed file: $LAST_DONE"
    
    if [ "$LAST_DONE" -ge "$END" ]; then
        echo "All files processed successfully!"
        break
    else
        echo "Continuing with next batch..."
        sleep 5
    fi
done

echo "Processing complete!"
# Optional: remove progress file when finished
# rm -f "$PROGRESS_FILE"
EOF

echo "All JSON Files added to the vector database."