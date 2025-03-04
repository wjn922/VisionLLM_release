#!/bin/bash

# mmcv, flash-attn, apex, deformable-attn, ops_dcnv3 need gpu

# install mmcv (v1.7.0)
# pip3 install -U openmim
pip3 install numpy==1.23.0  # <2.0
cd mmcv
export MMCV_WITH_OPS=1
pip3 install -v -e . 
cd ..

# install mmengine
pip3 install -U openmim
mim install mmengine

# install mmdet (v2.25.3)
# cd mmdetection
# pip3 install -v -e . --user
# cd ..

# install UNINEXT detectron2
cd UNINEXT
pip3 install -e . 
pip3 install shapely==1.7.1
pip3 install  jpeg4py visdom easydict scikit-image
pip3 install transformers tikzplotlib motmetrics
cd ..


# install packages
pip3 install -r requirements.txt

# install detectron2
# python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --user

# install tools
# pycocotools
# pip3 install pycocotools
# compatible with ytvos
pip3 install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI" 
# lvis
pip3 install lvis
# cityscape
pip3 install cityscapesScripts
# crowpose
cd crowdpose-api/PythonAPI/
python3 setup.py install
cd ../..

# flash-attn
python3 -m pip install ninja
python3 -m pip install flash-attn==2.3.3

# apex
cd apex
bash compile.sh
cd ..

# deformable-attn
cd visionllmv2/model/unipose/ops
python3 setup.py build install 
cd ../../../..

# ops dcnv3
cd visionllmv2/model/ops_dcnv3
python3 setup.py build install 
cd ../../..
