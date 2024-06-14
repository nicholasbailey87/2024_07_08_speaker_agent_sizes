#!/bin/bash

set -e

python merge.py --hostname cpu2 --logfile logs/log_20171108_191201cpu2.log --title 'embedding 10'
cp /tmp/out-reward.png images/emb10.png

python merge.py --hostname cpu2 --logfile logs/log_20171108_202808cpu1_emb50.log --title 'embedding 50'
cp /tmp/out-reward.png images/emb50.png

