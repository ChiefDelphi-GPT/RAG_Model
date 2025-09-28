# PowerShell script to clean up remote directory via SSH
# Equivalent to the bash script for Windows

# Set error action to stop on any error (equivalent to set -e)
$ErrorActionPreference = "Stop"

try {
    # Execute SSH command to remote server
    ssh fe.ds @'
set -e
cd /home/rhayrapetyan/automatic
rm -rf *
'@
    
    Write-Host "Remote cleanup completed. The automatic directory is now empty." -ForegroundColor Green
}
catch {
    Write-Error "Failed to execute remote cleanup: $_"
    exit 1
}