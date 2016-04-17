#!/bin/bash
git checkout dropout &&
cd deep_q_rl &&
python run_nips.py -r $1 --network-type $2 --max-history 100000
