#!/bin/bash
for i in {0..49};
do
    for DATASET in 'tweets' 'wikipedia_hate_speech' 'breitbart';
    do
        for BALANCE in .01 .05 .1 .3 .5; 
        do
            for MODE in 'random' 'active';
            do
                if [ $MODE == 'random' ];
                then
                    echo Bal: $BALANCE, Iter: $i, Mode: $MODE
                    python3 active_learning_sim.py $DATASET --mode $MODE --balance $BALANCE --iter $i
                else
                    for Q in 'committee' 'margin';
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
                            echo Bal: $BALANCE, Iter: $i, Mode: $MODE, Query Strat.: $Q
                            python3 active_learning_sim.py breitbart --mode $MODE --query_strat $Q --balance $BALANCE --iter $i
                        fi
                    done
                fi
            done
        done
    done
done
