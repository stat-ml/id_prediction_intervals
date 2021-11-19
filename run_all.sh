FILES="./configs/*.yaml"
METHODS='id dropout ens'
for m in $METHODS
do
  for f in $FILES
  do
    for c in {1..20}
    do
       python -m important_directions.experiment $f -m $m -s $c --gpu --data_gpu
    done
  done
done
