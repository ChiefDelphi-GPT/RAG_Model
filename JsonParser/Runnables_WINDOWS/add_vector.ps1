# run_vectors_resume.ps1
# Processes 0.json to 653.json, remembering where it left off.

$PROGRESS_FILE = "C:\Users\serge\Downloads\FRC\RAG_Model\JsonParser\Runnables_WINDOWS\progress_adding_vectors.txt"
$START = 0
$END = 1151

# If progress file exists, resume from the next index
if (Test-Path $PROGRESS_FILE) {
    $LAST_DONE = Get-Content $PROGRESS_FILE
    $START = [int]$LAST_DONE + 1
    Write-Host "Resuming from $START.json..."
} else {
    Write-Host "Starting fresh..."
}

for ($i = $START; $i -le $END; $i++) {
    Write-Host "Processing $i.json..."
    
    # Run the Python script
    python3 ../vector_info.py "../../Cheif_Delphi_JSONS/$i.json"
    
    # Check if the command was successful
    if ($LASTEXITCODE -eq 0) {
        # Save the current index after successful run (overwrite previous)
        $i | Out-File -FilePath $PROGRESS_FILE -Encoding UTF8

        # Print a confirmation message
        Write-Host "$i.json finished successfully." -ForegroundColor Green
    } else {
        Write-Host "Error on $i.json, stopping." -ForegroundColor Red
        exit 1
    }
}

Write-Host "All files processed." -ForegroundColor Green

# Optional: remove progress file when finished
if (Test-Path $PROGRESS_FILE) {
    Remove-Item $PROGRESS_FILE
}
