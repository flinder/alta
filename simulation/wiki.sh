#!/bin/bash
for MODE in 'random' 'active'; 
do
    for BALANCE in .01 .05 .1 .5;
    do
        for i in {10..15};
        do
            if [ $MODE='random' ];
            then
                echo Bal: $BALANCE, Iter: $i, Mode: $MODE
                python3 active_learning_sim.py wikipedia_hate_speech\
                    --mode $MODE --iter $i --balance $BALANCE
            else
                echo Bal: $BALANCE, Iter: $i, Mode: $MODE
                python3 active_learning_sim.py wikipedia_hate_speech\
                    --mode $MODE --iter $i --balance $BALANCE
            fi
        done
    done
done
