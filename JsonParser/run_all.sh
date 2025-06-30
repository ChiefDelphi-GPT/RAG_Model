#!/bin/bash
set -e

# Copy data directory to remote source
./clean_all.sh
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
cd JsonParser/

# Process files 3-149 first
for i in $(seq 3 149)
do
    python3 input_cleaner.py ../json_originals/${i}.json
done

# Then process files 0, 1, and 2
python3 input_cleaner.py ../json_originals/0.json
python3 input_cleaner.py ../json_originals/1.json
python3 input_cleaner.py ../json_originals/2.json
EORUN

chmod +x run.sh

# Submit job to Slurm
srun -p general --mem=800G --ntasks=2 -t 12:00:00 --gres=gpu:1 ./run.sh

exit
EOF

# Copy output directory back to local
scp -r fe.ds:/home/rhayrapetyan/automatic/Cheif_Delphi_JSONS /Users/rubenhayrapetyan/Downloads/Code/FRC/CheifDelphi-GPT/RAG_Model/

echo "All JSON files processed and copied to Cheif_Delphi_JSONS directory."