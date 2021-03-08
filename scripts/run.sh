#!/bin/bash
: '
----------------------------------------------------------------
Author: Jiho Choi
    - https://github.com/JihoChoi
----------------------------------------------------------------
'

# run step by step
# source ./env/bin/activate


# =======================
#     PREPARE DATASET
# =======================


echo "-----------------------------------"
echo "        DATASET PREPARATION        "
echo "-----------------------------------"

sh ./scripts/prepare_dataset.sh



# snapshot count: 3
python ./dynamic-gcn/preparation/preprocess_dataset.py Twitter16 3
python ./dynamic-gcn/preparation/prepare_snapshots.py Twitter16 sequential 3
python ./dynamic-gcn/preparation/prepare_snapshots.py Twitter16 temporal 3


echo "----------------------------"
echo "        TRAIN & TEST        "
echo "----------------------------"


python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter16 --dataset-type sequential --snapshot-num 3 \
    --cuda cuda:1

python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter16 --dataset-type temporal --snapshot-num 3 \
    --cuda cuda:1

python ./dynamic-gcn/main.py --model GCN --learning-sequence additive \
    --dataset-name Twitter16 --dataset-type temporal --snapshot-num 5 \
    --cuda cuda:1

python ./dynamic-gcn/main.py --model GCN --learning-sequence dot_product \
    --dataset-name Twitter16 --dataset-type temporal --snapshot-num 5 --cuda cuda:1
