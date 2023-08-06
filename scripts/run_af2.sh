#!/bin/bash

# predict protein structure for every folder in given path
# each folder may contain several .3m files

for file_path in ./output/casp15/orphan/A1T3R1.5/* # modify msa directory here
do
    if [ -d "$file_path" ]
    then   
        file_path=${file_path%/}/
        new_path=${file_path/output/af2}
        ./par-mcard-af2.sh $file_path $new_path
    fi
done

#  originial msa prediction
#  ./par-mcard-af2.sh ./dataset/casp15/cfdb ./af2/casp15/gold_label/ 
