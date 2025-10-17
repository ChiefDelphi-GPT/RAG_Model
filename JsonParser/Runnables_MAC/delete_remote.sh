#!/bin/bash
set -e

ssh fe.ds << 'EOF'
cd /home/rhayrapetyan/automatic
rm -f progress_clean_inputs.txt
rm -f progress_add_vectors.txt
rm -rf /home/rhayrapetyan/automatic/
EOF

echo "Remote cleanup completed. The automatic directory is now empty."