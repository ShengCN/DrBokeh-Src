#!/bin/bash
id=00007
file=Imgs/$id.png
alpha=Imgs/$id-alpha.png
alpha=Imgs/$id-alpha2.png
disp=Imgs/$id-disp.npz

# basename w. extension and w.o. extension
fbasename=$(basename "$file")
fbasename_wo_ext="${fbasename%.*}"

focal=0.1
python app/Render/Inference.py --rgb $file --alpha $alpha --disp $disp --K 30.0 --focal $focal --ofile outputs/$fbasename_wo_ext-focal-$focal.png --verbose --lens 71 --gamma 2.2

focal=0.8
python app/Render/Inference.py --rgb $file --alpha $alpha --disp $disp --K 30.0 --focal $focal --ofile outputs/$fbasename_wo_ext-focal-$focal.png --verbose --lens 71 --gamma 2.2