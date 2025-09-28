# remote_cleanup.ps1
# PowerShell equivalent of the bash SSH script

# Set error action preference to stop on errors (equivalent to set -e)
$ErrorActionPreference = "Stop"

# Execute SSH commands - combine into single command string
ssh fe.ds "set -e && cd /home/rhayrapetyan/automatic && rm -rf *"

Write-Host "Remote cleanup completed. The automatic directory is now empty." -ForegroundColor Green