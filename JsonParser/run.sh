#!/bin/bash

for i in {0..149}
do
    python3 json_parser.py ../Cheif_Delphi_JSONS/$i.json
done
