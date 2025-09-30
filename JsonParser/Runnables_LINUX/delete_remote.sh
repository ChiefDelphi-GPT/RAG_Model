#!/bin/bash
set -e

ssh fe.ds << 'EOF'
set -e
cd /home/rhayrapetyan/automatic
rm -rf *
EOF

echo "Remote cleanup completed. The automatic directory is now empty."