#!/bin/bash
for MODE in 'active'; #'random' 'active';
do
    for ICR in .7 .8 .85 .9 .95;
    do
        for BALANCE in .05 .1 .5 .01;
        do
            for i in {0..49};
            do
                if [ $MODE='random' ];
                then
                    echo Bal: $BALANCE, Iter: $i, Mode: $MODE, ICR: $ICR
                    python3 active_learning_sim.py tweets --random --balance $BALANCE --icr $ICR
                else
                    echo Bal: $BALANCE, Iter: $i, Mode: $MODE, ICR: $ICR
                    python3 active_learning_sim.py tweets --balance $BALANCE --icr $ICR
                fi
            done
        done
    done
done
