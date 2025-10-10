# run_vectors_resume_local_single.ps1
# Processes JSON files locally with progress tracking (one at a time)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# -------------------
# Configuration
# -------------------
$ProgressFile = Join-Path $PSScriptRoot "progress_clean_inputs.txt"
$Start        = 0
$End          = 1649

# Directories relative to this script
$InputDir  = Join-Path $PSScriptRoot "..\..\json_originals"
$OutputDir = Join-Path $PSScriptRoot "..\..\..\RAG_Model\Cheif_Delphi_JSONS"
$Cleaner   = Join-Path $PSScriptRoot "..\input_cleaner.py"

# -------------------
# Handle --fresh run
# -------------------
if ($args.Count -gt 0 -and $args[0] -eq "--fresh") {
    Write-Host "Starting a fresh run: deleting local progress file..."
    Remove-Item -Force $ProgressFile -ErrorAction SilentlyContinue
    Write-Host "Fresh run setup complete."
}

# -------------------
# Cleanup on Ctrl+C
# -------------------
$script:CurrentIndex = $Start
Register-EngineEvent PowerShell.Exiting -Action {
    Write-Host "`nCaught Ctrl+C! Saving progress at index $script:CurrentIndex"
    Set-Content -Path $ProgressFile -Value $script:CurrentIndex -Encoding UTF8
}

# -------------------
# Initialize / resume
# -------------------
if (-not (Test-Path $ProgressFile)) {
    Set-Content -Path $ProgressFile -Value $Start -Encoding UTF8
    Write-Host "Creating new progress file starting at $Start.json..."
} else {
    $LastDone = (Get-Content $ProgressFile -Raw).Trim()
    if ($LastDone) {
        $Start = [int]$LastDone
        Write-Host "Resuming from $Start.json..."
    }
}

# Ensure output directory exists
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# -------------------
# Main processing loop
# -------------------
for ($i = $Start; $i -le $End; $i++) {
    $script:CurrentIndex = $i
    $InputFile = Join-Path $InputDir "$i.json"

    if (Test-Path $InputFile) {
        Write-Host "Processing $i.json..."
        python $Cleaner $InputFile
        Set-Content -Path $ProgressFile -Value $i -Encoding UTF8
    } else {
        Write-Host "$i.json does not exist, skipping..."
    }
}

Write-Host "All JSON files processed into $OutputDir"
Remove-Item -Force $ProgressFile
