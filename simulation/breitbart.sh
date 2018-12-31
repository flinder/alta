#!/bin/bash
for i in {0..49};
do
    for BALANCE in .01 .05 .1 .3 .5; 
    do
        for Q in 'margin' 'committee';
        do
            for MODE in 'random' 'active'; 
            do
                echo Bal: $BALANCE, Iter: $i, Mode: $MODE
                python3 active_learning_sim.py breitbart --mode $MODE --query_strat $Q --balance $BALANCE --iter $i
            done
        done
    done
done
