#!/bin/bash

MAXTSTS=1

i=0

declare -a CITIES=( "Barcelona" "NYC" "London")
declare -a STAGES=( 4 2 3 )


for STAGE in "${STAGES[@]}" ;do
  for CITY in "${CITIES[@]}" ;do
    echo "$CITY"

    nohup /usr/bin/python3.6 -u  SemPicPoiCold.py -stg $STAGE -ct $CITY &

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

    #Esperar X segundos entre pruebas para que le de tiempo a ocupar memoria en GPU
    sleep 1 # 600

  done
done