#!/bin/bash
for i in {50..70};
do
    for DATASET in 'tweets';
    do
        for BALANCE in .01 .05 .1 .3 .5;
        do
            for MODE in 'active' 'random';
            do
                if [ $DATASET == 'tweets' ];
                then
                    echo Bal: $BALANCE, Iter: $i, Mode: $MODE
                    python3 active_learning_sim.py $DATASET --mode $MODE --balance $BALANCE --iter $i
                    for ICR in .7 .8 .85 .9 .95;
                    do
                        echo Bal: $BALANCE, Iter: $i, Mode: $MODE, ICR: $ICR
                        python3 active_learning_sim.py $DATASET --mode $MODE --balance $BALANCE --icr $ICR --iter $i
                    done
                else
                    echo Bal: $BALANCE, Iter: $i, Mode: $MODE
                    python3 active_learning_sim.py $DATASET --mode $MODE --balance $BALANCE --iter $i
                fi
            done
        done
    done
done