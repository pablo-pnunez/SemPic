#!/bin/bash

STAGE="train"
declare -a PLACES=("paris")
declare -a INDEXS=( 0 1 2 3 4 5 6 7 8 9)

#declare -a INDEXS=( 6 7 8 9 )
#declare -a INDEXS=( 10 11 12 13 14 15 16 17 18 19)

GPU=0

for WHERE in "${PLACES[@]}" ;do

    for IDX in "${INDEXS[@]}" ;do
        nohup /usr/bin/python3.6 -u  Semántica.py  -s $STAGE  -c "$WHERE" -i $IDX -gpu $GPU > "out/TR/${WHERE// /_}_"$IDX".txt" &
        GPU=$(($(($GPU+1%2))%2))
    done

done

