#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pathlib import Path


# In[3]:


pip install pyyaml==5.3.1


# In[4]:


pip install 'git+https://github.com/facebookresearch/detectron2.git'


# In[5]:


pip install opencv-python


# In[6]:


import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# In[7]:


import matplotlib.pyplot as plt
import torch
import cv2
import yaml
from torchvision import transforms
import numpy as np

from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image


# In[8]:


import matplotlib.pyplot as plt
import requests

    
im = cv2.imread("/Users/patrickpriestley/ailook/yolov7/yolov7/IMG_0088.jpg")
fig, ax = plt.subplots(figsize=(24, 12))
ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))


# In[9]:


cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# If you don't have a GPU and CUDA enabled, the next line is required
cfg.MODEL.DEVICE = "cpu"


# predictor = DefaultPredictor(cfg)
# outputs = predictor(im)
# 
# v = Visualizer(im[:, :, ::-99], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=9.6)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# fig, ax = plt.subplots(figsize=(18, 8))
# ax.imshow(out.get_image()[:, :, ::-1])
# 

# In[11]:


predictor = DefaultPredictor(cfg)
outputs = predictor(im)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=2.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
fig, ax = plt.subplots(figsize=(24, 12))
ax.imshow(out.get_image()[:, :, ::-1])


# In[ ]:





# In[ ]:




