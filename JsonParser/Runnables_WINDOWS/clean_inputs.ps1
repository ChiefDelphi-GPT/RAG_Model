# run_vectors_resume.ps1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# -------------------
# Configuration
# -------------------
$PROGRESS_FILE = Join-Path $PSScriptRoot "progress_clean_inputs.txt"
$START = 150
$END = 1151
$STEP = 3
$REMOTE_USER = "fe.ds"
$REMOTE_DIR = "/home/rhayrapetyan/automatic"
$REMOTE_JSON_DIR = "$REMOTE_DIR/Cheif_Delphi_JSONS"
$REMOTE_SCRIPT_DIR = "$REMOTE_DIR/JsonParser"
$ORIGINALS_DIR = "$REMOTE_DIR/json_originals_150-1151"

# -------------------
# Handle fresh run
# -------------------
if ($args.Count -gt 0 -and $args[0] -eq "--fresh") {
    Write-Host "Starting a fresh run: deleting local and remote progress files..."
    Remove-Item -Force $PROGRESS_FILE -ErrorAction SilentlyContinue
    ssh $REMOTE_USER "rm -f ${REMOTE_DIR}/progress_clean_inputs.txt"
    Write-Host "Fresh run setup complete."
}

# -------------------
# Retry SCP function
# -------------------
function Retry-SCP {
    param([string]$Src, [string]$Dest)
    $Delay = 5
    while ($true) {
        try {
            scp -r $Src $Dest
            Write-Host "Successfully copied $Src to $Dest"
            break
        } catch {
            Write-Host "Copy failed for $Src → $Dest, retrying in ${Delay}s..."
            Start-Sleep -Seconds $Delay
        }
    }
}

# -------------------
# Cleanup on Ctrl+C
# -------------------
function Cleanup-And-Exit {
    Write-Host "`nCaught Ctrl+C! Updating local progress before exiting..."
    Retry-SCP "${REMOTE_USER}:${REMOTE_DIR}/progress_clean_inputs.txt" $PROGRESS_FILE
    try {
        Retry-SCP "${REMOTE_USER}:${REMOTE_JSON_DIR}/*" (Join-Path $PSScriptRoot "../../../RAG_Model/Cheif_Delphi_JSONS/")
    } catch {
        Write-Host "No JSON files to copy back, continuing..."
    }
    Write-Host "Exiting..."
    exit 130
}
$null = Register-EngineEvent PowerShell.Exiting -Action { Cleanup-And-Exit }

# -------------------
# Initialize local progress
# -------------------
if (-Not (Test-Path $PROGRESS_FILE)) {
    Set-Content -Path $PROGRESS_FILE -Value $START -Encoding UTF8
    Write-Host "Creating new progress file starting at $START.json..."
}

# Copy progress file to remote
scp $PROGRESS_FILE "${REMOTE_USER}:${REMOTE_DIR}/"

# Determine last done locally
$LAST_DONE = (Get-Content $PROGRESS_FILE -Raw).Trim()
if ($LAST_DONE -ne "") {
    $START = [int]$LAST_DONE
    Write-Host "Resuming from $START.json..."
} else {
    Write-Host "Starting fresh..."
}

# -------------------
# Copy data and scripts to remote
# -------------------
& "$PSScriptRoot\delete_remote.ps1"

scp -r (Join-Path $PSScriptRoot "../../json_originals_150-1151") "${REMOTE_USER}:${REMOTE_DIR}/"
ssh $REMOTE_USER "mkdir -p ${REMOTE_SCRIPT_DIR} ${REMOTE_JSON_DIR}"

scp (Join-Path $PSScriptRoot "../input_cleaner.py") "${REMOTE_USER}:${REMOTE_SCRIPT_DIR}/"
scp $PROGRESS_FILE "${REMOTE_USER}:${REMOTE_DIR}/" -ErrorAction SilentlyContinue

# -------------------
# SSH to remote and run processing
# -------------------
$remoteBash = @'
#!/bin/bash
set -e

PROGRESS_FILE="/home/rhayrapetyan/automatic/progress_clean_inputs.txt"
SCRIPT_DIR="/home/rhayrapetyan/automatic/JsonParser"
ORIGINALS_DIR="/home/rhayrapetyan/automatic/json_originals_150-1151"
JSON_DIR="/home/rhayrapetyan/automatic/Cheif_Delphi_JSONS"

mkdir -p "$SCRIPT_DIR" "$JSON_DIR"

if [[ -f "$PROGRESS_FILE" ]]; then
    START=$(cat "$PROGRESS_FILE")
else
    START=150
fi
END=1151
STEP=3

cd "$SCRIPT_DIR"

cat > run.sh <<EORUN
#!/bin/bash
#!/bin/bash
PROGRESS_FILE="/home/rhayrapetyan/automatic/progress_clean_inputs.txt"
for i in $(seq $START $STEP $END); do
    srun -p general --mem=800G --ntasks=1 -t 6:00:00 --gres=gpu:1 \
         bash -c "python3 input_cleaner.py ../json_originals_150-1151/\${i}.json && \
                  echo \${i} > \$PROGRESS_FILE && \
                  if [ -f ../json_originals_150-1151/\$((i+1)).json ]; then
                      python3 input_cleaner.py ../json_originals_150-1151/\$((i+1)).json && echo \$((i+1)) > \$PROGRESS_FILE
                  fi
                  if [ -f ../json_originals_150-1151/\$((i+2)).json ]; then
                      python3 input_cleaner.py ../json_originals_150-1151/\$((i+2)).json && echo \$((i+2)) > \$PROGRESS_FILE
                  fi"
done
EORUN

sed -i 's/\r$//' run.sh
chmod +x run.sh
./run.sh
'@

# Remove CR characters from the PowerShell string
$remoteBash = $remoteBash -replace "`r",""

# Execute the SSH remote script
ssh $REMOTE_USER $remoteBash

# -------------------
# Copy processed files back to local
# -------------------
try {
    Retry-SCP "${REMOTE_USER}:${REMOTE_JSON_DIR}/*" (Join-Path $PSScriptRoot "../../../RAG_Model/Cheif_Delphi_JSONS/")
} catch {
    Write-Host "No JSON files found on remote, continuing..."
}

Write-Host "All JSON files processed and copied to Cheif_Delphi_JSONS directory."
Remove-Item -Force $PROGRESS_FILE
