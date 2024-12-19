#!/bin/bash

SEED=$(date +%s)
Y=100 # Total amount of random prompts

R=$(($SEED % Y+1))
P=$(sed -n "${R}p" prompts.txt)
echo "Randomized prompt: $P"

echo "Doing Face Generation..."
bash generate.sh "$P" > ./gen.log

DIRECTORY="../GenTmp/samples"
NUM_IMG=6
new_images=$(ls -1 "$DIRECTORY" | sort | tail -n "$NUM_IMG")

echo "Doing Face Swap..."

i=0
for file in $new_images
do
    bash swap.sh "../$DIRECTORY/$file" $1 > ./swp_$i.log &
    i=$((i+1))
done

wait

echo "Doing Evaluation..."
bash evaluate.sh $1

echo "Done!"