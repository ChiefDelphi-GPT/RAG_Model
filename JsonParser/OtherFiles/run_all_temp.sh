#!/bin/bash
set -e

ssh fe.ds << 'EOF'
set -e
cd /home/rhayrapetyan/automatic/

# Create the run.sh script
cat > run.sh << 'EORUN'
#!/bin/bash
cd JsonParser/

for i in $(seq 117 149)
do
    srun -p general --mem=800G --ntasks=2 -t 2:00:00 --gres=gpu:1 python3 input_cleaner.py ../json_originals/${i}.json
done
EORUN

chmod +x run.sh

./run.sh

exit
EOF

scp -r fe.ds:/home/rhayrapetyan/automatic/Cheif_Delphi_JSONS /Users/rubenhayrapetyan/Downloads/Code/FRC/CheifDelphi-GPT/RAG_Model/

echo "All JSON files processed and copied to Cheif_Delphi_JSONS directory."