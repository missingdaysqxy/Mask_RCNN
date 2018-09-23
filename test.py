# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 17:03:32 2018

@author: qxliu
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

#自定义本次测试使用的模型名称
MODEL_NAME = "mask_rcnn-gxl.h5"

#根目录
ROOT_DIR = os.path.abspath(".")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
print('工作目录为：%s' % ROOT_DIR)

# Local path to trained weights file
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_SAVE_DIR = os.path.join(ROOT_DIR, "SavedModels")
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
assert os.path.exists(MODEL_PATH)==True, "测试失败：" + MODEL_SAVE_DIR + "下未找到" + MODEL_NAME

class InferenceConfig(Config):
    # Give the configuration a recognizable name
    NAME = "persons"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + count of class
    
    # 虽然图片很大，但图中物体较小，因此使用较小的候选区
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    IMAGE_MIN_DIM = 720
    IMAGE_MAX_DIM = 1280
    IMAGE_RESIZE_MODE = "square"

    TRAIN_ROIS_PER_IMAGE = 200
    STEPS_PER_EPOCH = 200
    VALIDATION_STEPS = 25

inference_config = InferenceConfig()
inference_config.display()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Load trained weights
print("Loading weights from ", MODEL_PATH)
model.load_weights(MODEL_PATH, by_name=True)
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ["BG","person","laptop","pen","bottle"]

# 读取测试图片
test_img_folder = os.path.join(ROOT_DIR, "test_img")   #存放测试图像
test_result = os.path.join(ROOT_DIR, "test_result")   #存放测试结果

img_exts= ['.jpg','.jpeg','.png','.gif','.bmp']
for root, dirs, files in os.walk(test_img_folder):
    for name in files:
        if os.path.splitext(name)[1].lower() in img_exts:
            image_path = os.path.join(root,name)
            print(image_path)
            image = skimage.io.imread(image_path)
            # Run detection
            results = model.detect([image], verbose=1)
            # Visualize results
            r = results[0]
            print(r['scores'])
            num=len(r['class_ids'])
            masked_img = visualize.display_instances(image, r['rois'],
                        r['masks'], r['class_ids'], class_names, r['scores'])
            print(num)
            save_path = root.replace(test_img_folder, test_result)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, name)
            save_path = os.path.splitext(save_path)[0]+'_masked.png'
            skimage.io.imsave(save_path,masked_img)
            print("Masked image was saved as " + save_path)
