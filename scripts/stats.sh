#!/bin/bash

MAXTSTS=1
i=0

declare -a CITIES=( "gijon" "barcelona" "madrid" "newyorkcity" "paris" "london")
declare -a CITIES=( "London")

for CITY in "${CITIES[@]}" ;do
  echo "$CITY"
  nohup /usr/bin/python3.6 -u  SemPicPoi.py -ct $CITY &

  # Almacenar los PID en una lista hasta alcanzar el máximo de procesos
  pids[${i}]=$!
  i+=1

  echo "   -[$!] $CITY"

  # Si se alcanza el máximo de procesos simultaneos, esperar
  if [ "${#pids[@]}" -eq $MAXTSTS ];
  then

    # Esperar a que acaben los X
    for pid in ${pids[*]}; do
        wait $pid
    done
    pids=()
    i=0
  fi


done