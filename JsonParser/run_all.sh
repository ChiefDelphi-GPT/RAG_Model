#!/bin/bash
set -e

# Copy data directory to remote
scp -r /Users/rubenhayrapetyan/Downloads/Code/FRC/CheifDelphi-GPT/RAG_Model/json_originals fe.ds:/home/rhayrapetyan/automatic/
scp -r /Users/rubenhayrapetyan/Downloads/Code/FRC/CheifDelphi-GPT/RAG_Model/JsonParser fe.ds:/home/rhayrapetyan/automatic/


# SSH and execute commands on remote
ssh fe.ds << 'EOF'
set -e
cd /home/rhayrapetyan/automatic/
mkdir -p /home/rhayrapetyan/automatic/Cheif_Delphi_JSONS
# Create the run.sh script
cat > run.sh << 'EORUN'
#!/bin/bash
cd /home/rhayrapetyan/automatic/JsonParser  # ADD THIS LINE
for i in {0..149}
do
    python3 input_cleaner.py ../json_originals/${i}.json  # USE RELATIVE PATH
done
EORUN

chmod +x run.sh

# Submit job to Slurm
srun -p general --mem=500G --ntasks=2 -t 10:00:00 --gres=gpu:1 ./run.sh

exit
EOF

# Copy output directory back to local
scp -r fe.ds:/home/rhayrapetyan/automatic/Cheif_Delphi_JSONS /Users/rubenhayrapetyan/Downloads/Code/FRC/CheifDelphi-GPT/RAG_Model/

echo "All JSON files processed and copied to Cheif_Delphi_JSONS directory."
