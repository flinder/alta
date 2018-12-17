#!/bin/bash
for i in {30..49};
do
	for BALANCE in .01 .05 .1; 
    do
    	for PCTRAND in 0.0 .25 .5 .75;
    	do
        	echo Bal: $BALANCE, Iter: $i, Pct Rand: $PCTRAND
			python3 active_learning_sim.py tweets\
				--balance $BALANCE --query_strat margin\
				--mode active --iter $i --pct_random $PCTRAND
    	done
    done
done
