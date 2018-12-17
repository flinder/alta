#!/bin/bash
for i in {0..49};
do
	for BALANCE in .05 .1; 
    do
        echo Bal: $BALANCE, Iter: $i, Mode: active
        python3 active_learning_sim.py tweets\
            --balance $BALANCE --query_strat committee --mode active --iter $i
    done
done
