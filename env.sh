pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# pip install pandas tqdm matplotlib pyyaml opencv-python easydict scikit-learn scikit-image pytorch-lightning kornia hydra-core albumentations==0.5.2  webdataset 
pip install -r requirements.txt

cd app/cuda-src && pip install . && cd -
