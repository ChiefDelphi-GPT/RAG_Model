#!/bin/bash

for i in {0..149}
do
    python3 input_cleaner.py ../Cheif_Delphi_JSONS/$i.json
done
