#!/bin/bash
for i in {0..49};
do
    for BALANCE in .01 .05 .1 .3 .5; 
    do
        for MODE in 'active'; 
        do
            for Q in 'committee';
            do
                echo Bal: $BALANCE, Iter: $i, Mode: $MODE, Query Strat.: $Q
                python3 active_learning_sim.py breitbart --mode $MODE --query_strat $Q --balance $BALANCE --iter $i
            done
        done
    done
done
