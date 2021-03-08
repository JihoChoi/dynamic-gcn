#!/bin/bash

# =======================
#     PREPARE DATASET
# =======================

: '
----------------------------------------------------------------
    Author: Jiho Choi
        - https://github.com/JihoChoi

    References
    - Detect Rumors in Microblog Posts Using Propagation Structure via Kernel Learning (ACL 2017)
    - Rumor Detection on Twitter with Tree-structured Recursive Neural Networks (ACL 2018)
        - Papers    : https://aclweb.org/anthology/papers/P/P17/P17-1066/
                    : https://aclweb.org/anthology/papers/P/P18/P18-1184/
        - Dataset   : https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0
----------------------------------------------------------------
'

echo "\n"
echo "-----------------------"
echo "        PREPARE        "
echo "-----------------------\n"

if [ ! -d ./data/ ]; then
    echo "CREATE [./data] directory"
    mkdir ./data/
fi

if [ ! -d ./data/raw/ ]; then
    echo "CREATE [./data/raw] directory"
    mkdir ./data/raw/
fi

if [ -d ./data/raw/rumor_detection_acl2017 ]; then
    echo "REMOVE ./data/raw/rumor_detection_acl2017 directory"
    rm -rf ./data/raw/rumor_detection_acl2017  # remove raw directory
fi

echo "\n"
echo "------------------------"
echo "        DOWNLOAD        "
echo "------------------------\n"
wget -O ./data/raw/rumdetect2017.zip https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip\?dl\=1

echo "\n"
echo "---------------------"
echo "        UNZIP        "
echo "---------------------\n"
# data
unzip -q data/raw/rumdetect2017.zip -d data/raw/

# resource
unzip ./resources/Weibo/weibotree.txt.zip -d ./resources/Weibo/
