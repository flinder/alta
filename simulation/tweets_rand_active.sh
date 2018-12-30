#!/bin/bash
for i in 49;
do
	for BALANCE in .01 .05 .1; 
    do
    	for PCTRAND in 0.0 0.25 0.75;
    	do
        	echo Bal: $BALANCE, Iter: $i, Pct Rand: $PCTRAND
			gtimeout 90m python3 active_learning_sim.py tweets\
				--balance $BALANCE --query_strat margin\
				--mode active --iter $i --pct_random $PCTRAND
    	done
    done
done
