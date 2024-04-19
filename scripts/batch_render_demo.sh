#!/bin/bash

# Input RGBs
files=("Imgs/00000.png" "Imgs/00003.png" "Imgs/00007.png" "Imgs/00012.png" "Imgs/00017.png" "Imgs/00019.png" "Imgs/00022.png")

# focal plane parameters 
focals=(0.2 0.8)

# Loop over each file
for file in "${files[@]}"; do
    # Loop over each parameter
    for focal in "${focals[@]}"; do
        # Execute a Python script with the current file and parameter
        echo "Running DrBokeh with file $file and param $focal"
		fbasename=$(basename "$file")
		fbasename_wo_ext="${fbasename%.*}"
		python app/Render/Inference.py --rgb $file --K 30.0 --focal $focal --ofile outputs/$fbasename_wo_ext-focal-$focal.png --lens 71 --gamma 2.2 
    done
done