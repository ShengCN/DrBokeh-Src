# Dr. Bokeh: DiffeRentiable Occlusion-aware Bokeh Rendering

This is the source code ([[**Bokeh Render**](#render)], [[**Depth-from-Defocus**](#depth-from-defocus)]) for Dr. Bokeh.

## Updates
-  [-] Src for Bokeh Render Demo   
	- [x] Compile the Dr.Bokeh
	- [x] Salient Detection Model  
	- [x] Inpainting model   
	- [x] Render demo 
	- [x] Documentation 
	- [ ] Docker image

-  [ ] Differentiable for depth-from-defocus   
	- [ ] Compile the Dr.Bokeh
	- [ ] Demo of differentiable of one pair 
	- [ ] Demo of differentiable of a dataset

- [ ] (**Important**) Record a video to explain the paper and code details. 

## Environment prerequisite 
First make sure you have ``nvcc==11.7`` installed. Other cuda version should also be OK, but not fully be tested.

Run the following bash to setup environment.

```bash
conda create -n drbokeh python=3.9 -y
conda activate py39
bash env.sh
```

### Setup inpainting & salient detection models
Dr. Bokeh relies on a layerred representation that can be obtained by matting + inpainting. 
Dr. Bokeh assumes high quality matting + inpainting as it is only responsible for rendering part.
So the matting and inpainting quality will affect Dr. Bokeh results.
In the paper and this code repo, we use [LDF](https://github.com/weijun88/LDF) for salient object detection and [lama inpainting](https://github.com/advimman/lama) for RGB inpainting, [MiDas](https://github.com/isl-org/MiDaS) for monocular depth.
For simplicity, this code repo does not use submodule. 
We hardcopy the codebases to this repo to make life a little happier.

Download the LDF weight from [here](https://huggingface.co/ysheng/DrBokeh/resolve/main/resnet50-19c8e357.pth?download=true) and put the weight to ``app/Render/Salient/LDF/res``.

Download the lama weight from [here](https://huggingface.co/ysheng/DrBokeh/resolve/main/best.ckpt?download=true) and put the weight to ``app/Render/Inpainting/lama/big-lama/models``.

Download the MiDas weight from [here](https://huggingface.co/ysheng/DrBokeh/resolve/main/dpt_large-midas-2f21e586.pt?download=true) and put the weight to ``app/Render/Depth/DPT/weights``

Or use the following code from terminal: 
```bash
wget -O app/Render/Salient/LDF/res/resnet50-19c8e357.pth https://huggingface.co/ysheng/DrBokeh/resolve/main/resnet50-19c8e357.pth?download=true

wget -O app/Render/Inpainting/lama/big-lama/models/best.ckpt https://huggingface.co/ysheng/DrBokeh/resolve/main/best.ckpt?download=true

wget -O app/Render/Depth/DPT/weights/dpt_large-midas-2f21e586.pt https://huggingface.co/ysheng/DrBokeh/resolve/main/dpt_large-midas-2f21e586.pt?download=true

```

### [Optional] Docker environment


## Render
Dr.Bokeh takes RGB, depth (for disparity), matting (alpha map for layerred representation) as inputs. 
If you do not provide depth or matting, the code will use off-the-shelf models discussed above to obtain a layerred representation. If you find the default off-the-shelf models are not good, you can replace them freely with any more advanced models.  
In practice, I notice [remove.bg](https://remove.bg) is good enough for matting most of the case.

We provide two scripts to illustrate how to use provided rendering scripts: ``scripts/batch_render_demo.sh`` and ``scripts/render_demo.sh``. Some testing images are put in the Imgs. 

``scripts/batch_render_demo.sh`` has the following codes: 
```bash
#!/bin/bash

# Input RGBs
files=("Imgs/00000.png" "Imgs/00003.png" "Imgs/00007.png" "Imgs/00012.png" "Imgs/00017.png" "Imgs/00019.png" "Imgs/00022.png")

# focal plane parameters 
focals=(0.2 0.8)

# Loop over each file
for file in "${files[@]}"; do
    # Loop over each parameter
    for focal in "${focals[@]}"; do
        echo "Running DrBokeh with file $file and param $focal"
		fbasename=$(basename "$file")
		fbasename_wo_ext="${fbasename%.*}"
		python app/Render/Inference.py --rgb $file --K 30.0 --focal $focal --ofile outputs/$fbasename_wo_ext-focal-$focal.png --lens 71 --gamma 2.2 
    done
done
```
It iterates over `files` and focus on ``0.2 and 0.8`` (disparity is normalized into [0.0,1.0] range) planes, and call Dr.Bokeh to render the results using default matting/depth/inpainting methods.

``scripts/render_demo.sh`` has the following codes: 
```bash
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
```

It renders bokeh result for one RGB image. It further shows how to use your own alpha/disp as inputs. Note, alpha assumes RGBA input with alpha map in the last channel. disp assumes you save your depth in a compressed npz format with `data` as the key. Details can be found in the code.
You can try to use different alpha inputs provided in the Imgs to see the result difference. 


Other rendering parameters also affect the results/performances. Here is the list of parameters for ``Inference.py``

```bash
usage: Inference.py [-h] --rgb RGB -K K_BLUR --focal FOCAL --ofile OFILE [--verbose] [--lens LENS] [--disp DISP] [--alpha ALPHA] [--gamma GAMMA]

DScatter Inference

optional arguments:
  -h, --help            show this help message and exit
  --rgb RGB             RGB input
  -K K_BLUR, --K_blur K_BLUR
                        Blur strength, a scaling factor that affects how large the bokeh shape would be
  --focal FOCAL         Focal plane
  --ofile OFILE         Bokeh output path
  --verbose             Verbose mode, will save the middle image representation for debug
  --lens LENS           Biggest lens kernel size, i.e. the largest neighborhood region
  --disp DISP           [Optional] Disparity input. Better disparity map makes results better. If not given, we use DPT to predict the depth.
  --alpha ALPHA         [Optional] Alpha map input. Better alpha map makes results better. If not given, we use LDF to segment the salient object.
  --gamma GAMMA         Naive gamma correction
```
``-K`` will affect the blur strength, it is a scaling factor meaning the blur radius of a pixel with defocus disparity of 1.0. 

``--lens`` will affect the neighborhood Dr.Bokeh will search. Larger lens allows larger blur strength, but takes more time for computation as a cost. 

``--gamma`` will affect the bokeh highlight intensity.  



## Depth-from-Defocus
TODO 
