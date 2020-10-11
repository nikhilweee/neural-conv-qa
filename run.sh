#! /bin/bash

echo "Activating environment."
eval "$(conda shell.bash hook)"
conda activate py36
echo "Enable Debugging."
set -ex


echo "CREATING DATASETS"


echo "Copying dataset files."
mkdir data/
cp sharc1-official/json/sharc_dev.json data/
cp sharc1-official/json/sharc_train.json data/

echo "Create Multiple References"
python3 scripts/create_multi_ref.py --dev-in data/sharc_dev.json --dev-out data/sharc_dev_multi.json


echo "Creating History Dataset"
python3 scripts/create_history_shuffled.py \
    --train-in data/sharc_train.json --dev-in data/sharc_dev.json \
    --train-out data/history_train.json --dev-out data/history_dev.json

echo "Create Multiple References"
python3 scripts/create_multi_ref.py --dev-in data/history_dev.json --dev-out data/history_dev_multi.json


echo "Creating New Dataset"
python3 scripts/create_sharc_mod.py \
    --train-in data/sharc_train.json --dev-in data/sharc_dev.json \
    --train-out data/mod_train.json --dev-out data/mod_dev.json

echo "Create Multiple References"
python3 scripts/create_multi_ref.py --dev-in data/mod_dev.json --dev-out data/mod_dev_multi.json


echo "EVALUATING"

python3 scripts/heuristic.py data/sharc_dev.json
python3 scripts/heuristic.py --multi data/sharc_dev_multi.json 
python3 scripts/heuristic.py --multi data/history_dev_multi.json 
python3 scripts/heuristic.py --multi data/mod_dev_multi.json 

echo "Done."