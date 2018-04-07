#!/bin/bash
for MODE in 'random' 'active'; 
do
    for BALANCE in .01 .05 .1 .5; 
    do
        for i in {0..49};
        do
            if [ $MODE='random' ];
            then
                echo Bal: $BALANCE, Iter: $i, Mode: $MODE
                python3 active_learning_sim.py tweets\ 
                    --random --balance $BALANCE
            else
                echo Bal: $BALANCE, Iter: $i, Mode: $MODE
                python3 active_learning_sim.py tweets\
                    --balance $BALANCE
            fi
        done
    done
done
