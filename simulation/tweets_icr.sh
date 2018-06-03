#!/bin/bash
for i in {0..49};
do
    for ICR in .7 .8 .85 .9 .95;
    do
        for BALANCE in .05 .1 .5 .01;
        do
            for MODE in 'active' 'random';
            do
                echo Bal: $BALANCE, Iter: $i, Mode: $MODE, ICR: $ICR
                python3 active_learning_sim.py tweets --mode $MODE --balance $BALANCE --icr $ICR --iter $i
            done
        done
    done
done