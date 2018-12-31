#!/bin/bash
for i in {0..49};
do
    for ICR in .7 .8 .85 .9 .95;
    do
        for BALANCE in .01 .05 .1 .3 .5;
        do
            for MODE in 'active';
            do
                echo Bal: $BALANCE, Iter: $i, Mode: $MODE, ICR: $ICR
                python3 active_learning_sim.py tweets --mode $MODE --query_strat margin --balance $BALANCE --icr $ICR --iter $i
            done
        done
    done
done