# Dr. Bokeh: DiffeRentiable Occlusion-aware Bokeh Rendering

This is the source code ([[**Bokeh Render**](#Render)], [[**Depth-from-Defocus**](Depth-from-Defocus)]) for Dr. Bokeh.

## Updates
-  [-] Src for Bokeh Render Demo   
	- [x] Compile the Dr.Bokeh
	- [x] Salient Detection Model  
	- [x] Inpainting model   
	- [x] Render demo 
	- [ ] Documentation 

-  [ ] Differentiable for depth-from-defocus   
	- [ ] Compile the Dr.Bokeh
	- [ ] Demo of differentiable of one pair 
	- [ ] Demo of differentiable of a dataset

- [ ] Colab? 
- [ ] Docker image is needed

## Environment prerequisite 
First make sure you have ``nvcc==11.7`` installed. Other cuda version should also be OK, but not fully be tested.

Run the following bash to setup environment.

```bash
conda create -n drbokeh py39 
conda activate py39

bash env.sh
```

### Setup inpainting & salient detection models
Dr. Bokeh relies on a layerred representation that can be obtained by matting + inpainting. 
Dr. Bokeh assumes high quality matting + inpainting as it is only responsible for rendering part.
So the matting and inpainting quality will affect Dr. Bokeh results.
In the paper and this code repo, we use [LDF](https://github.com/weijun88/LDF) for salient object detection and [lama inpainting](https://github.com/advimman/lama) for RGB inpainting.
For simplicity, this code repo does not use submodule. 
We hardcopy one version in this repo.

To setup the environment, download the LDF weight [here]() and put the weight to 

### [Optional] Docker environment


## Render



## Depth-from-Defocus

