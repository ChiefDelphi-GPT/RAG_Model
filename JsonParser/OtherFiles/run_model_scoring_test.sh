#!/bin/bash
set -e

# Copy data directory to remote source
scp /Users/rubenhayrapetyan/Downloads/Code/FRC/CheifDelphi-GPT/RAG_Model/JsonParser/model_scoring_test.py fe.ds:/home/rhayrapetyan/automatic/JsonParser/

# SSH and execute commands on remote
ssh fe.ds << 'EOF'
set -e
cd /home/rhayrapetyan/automatic/JsonParser
srun -p general --mem=800G --ntasks=2 -t 6:00:00 --gres=gpu:1 python3 model_scoring_test.py
exit
EOF

