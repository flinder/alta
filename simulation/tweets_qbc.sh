#!/bin/bash
for i in {0..49};
do
	for BALANCE in .01 .05 .1 .3 .5; 
    do
        echo Bal: $BALANCE, Iter: $i, Mode: active
        python3 active_learning_sim.py tweets\
            --balance $BALANCE --query_strat committee --mode active --iter $i
    done
done
