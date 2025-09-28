# PowerShell equivalent of the bash script
param(
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"

$PROGRESS_FILE = "progress_clean_inputs.txt"
$START = 150

if ($Fresh) {
    Write-Host "Starting a fresh run: deleting local and remote progress files..."
    if (Test-Path $PROGRESS_FILE) {
        Remove-Item $PROGRESS_FILE -Force
    }
    ssh fe.ds "rm -f /home/rhayrapetyan/automatic/progress_clean_inputs.txt"
    Write-Host "Fresh run setup complete."
}

function Invoke-ScpRetry {
    param(
        [string]$Source,
        [string]$Destination
    )
    
    $delay = 5
    while ($true) {
        try {
            scp -r $Source $Destination
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Successfully copied $Source to $Destination"
                break
            } else {
                throw "SCP failed with exit code $LASTEXITCODE"
            }
        } catch {
            Write-Host "Copy failed for $Source → $Destination, retrying in ${delay}s..."
            Start-Sleep -Seconds $delay
        }
    }
}

function Invoke-CleanupAndExit {
    Write-Host ""
    Write-Host "Caught Ctrl+C! Updating local progress before exiting..."

    # Keep trying to copy the remote progress file
    Invoke-ScpRetry "fe.ds:/home/rhayrapetyan/automatic/progress_clean_inputs.txt" $PROGRESS_FILE

    # Keep trying to copy all processed JSON files
    Invoke-ScpRetry "fe.ds:/home/rhayrapetyan/automatic/Cheif_Delphi_JSONS/*" "../../../RAG_Model/Cheif_Delphi_JSONS/"

    Write-Host "Exiting..."
    exit 130
}

# Set up trap to catch Ctrl+C
Register-EngineEvent PowerShell.Exiting -Action { Invoke-CleanupAndExit }
$null = Register-ObjectEvent -InputObject ([Console]) -EventName "CancelKeyPress" -Action { Invoke-CleanupAndExit }

if (-not (Test-Path $PROGRESS_FILE)) {
    Set-Content -Path $PROGRESS_FILE -Value $START
    Write-Host "Creating new progress file starting at $START.json..."
}

if (Test-Path $PROGRESS_FILE) {
    scp $PROGRESS_FILE fe.ds:/home/rhayrapetyan/automatic/
}

if (Test-Path $PROGRESS_FILE) {
    $LAST_DONE = Get-Content $PROGRESS_FILE
    $START = [int]$LAST_DONE
    Write-Host "Resuming from $START.json..."
} else {
    Write-Host "Starting fresh..."
}

# Copy data directory to remote source
& "./delete_remote.ps1"
scp -r ../../json_originals_150-1649 fe.ds:/home/rhayrapetyan/automatic/
ssh fe.ds "mkdir -p /home/rhayrapetyan/automatic/JsonParser"
scp ../input_cleaner.py fe.ds:/home/rhayrapetyan/automatic/JsonParser/
try {
    scp $PROGRESS_FILE fe.ds:/home/rhayrapetyan/automatic/
} catch {
    # Ignore error (equivalent to || true)
}

# SSH and execute commands on remote
$sshScript = @'
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

END=1650
STEP=3

cd /home/rhayrapetyan/automatic/
mkdir -p /home/rhayrapetyan/automatic/Cheif_Delphi_JSONS

# Create the run.sh script with proper escaping
cat > run.sh << 'EORUN'
#!/bin/bash
cd JsonParser/

START=150
END=1650
STEP=3

PROGRESS_FILE="/home/rhayrapetyan/automatic/progress_clean_inputs.txt"
if [ -f "$PROGRESS_FILE" ]; then
    LAST_DONE=$(cat "$PROGRESS_FILE")
    START=$((LAST_DONE))
fi

for i in $(seq $START $STEP $END); do
    srun -p general --mem=800G --ntasks=1 -t 6:00:00 --gres=gpu:1 \
        bash -c "
            python3 input_cleaner.py ../json_originals_150-1649/${i}.json && echo ${i} > $PROGRESS_FILE

            if [ -f ../json_originals_150-1649/\$((${i} + 1)).json ]; then
                python3 input_cleaner.py ../json_originals_150-1649/\$((${i} + 1)).json && echo \$((${i} + 1)) > $PROGRESS_FILE
            fi

            if [ -f ../json_originals_150-1649/\$((${i} + 2)).json ]; then
                python3 input_cleaner.py ../json_originals_150-1649/\$((${i} + 2)).json && echo \$((${i} + 2)) > $PROGRESS_FILE
            fi
        "
done
EORUN

chmod +x run.sh

# Run the processing script
./run.sh

exit
'@

ssh fe.ds $sshScript

# Copy output directory back to local
scp -r 'fe.ds:/home/rhayrapetyan/automatic/Cheif_Delphi_JSONS/*' ../../../RAG_Model/Cheif_Delphi_JSONS/

Write-Host "All JSON files processed and copied to Cheif_Delphi_JSONS directory."
if (Test-Path $PROGRESS_FILE) {
    Remove-Item $PROGRESS_FILE -Force
}