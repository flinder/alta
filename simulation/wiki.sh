#!/bin/bash
for i in {0..49};
do
    for BALANCE in .01 .05 .1 .5;
    do
        for Q in 'margin' 'committee';
        do
            echo Bal: $BALANCE, Iter: $i, Mode: $MODE
            python3 active_learning_sim.py wikipedia_hate_speech\
                    --mode active --iter $i --balance $BALANCE --query_strat $Q
        done
    done
done
