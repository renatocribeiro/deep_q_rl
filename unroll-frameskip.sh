#!/bin/bash
git checkout unroll-frameskip &&
cd deep_q_rl &&
python run_nips.py -r $1 --network-type just_ram --max-history 100000
