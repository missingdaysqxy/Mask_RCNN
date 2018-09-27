# -*- coding: utf-8 -*-

# In[1]:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml

# In[2]:
#根目录
ROOT_DIR = os.path.abspath(".")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
print('工作目录为：%s' % ROOT_DIR)
# 路径设置
def get_dataset_list(dataset_folder):
    list = os.listdir(dataset_folder)  # 该文件夹下所有的文件与子文件夹名称
    imglist=[]
    for name in list:
        if name.endswith('_json'):
            if os.path.exists(os.path.join(dataset_folder, name + '/label.png')) and\
               os.path.exists(os.path.join(dataset_folder, name + '/img.png')) and\
               os.path.exists(os.path.join(dataset_folder, name + '/info.yaml')):
                   imglist.append(name)
    return imglist
dataset_folder = os.path.join(ROOT_DIR, "dataset")   #训练数据目录
imglist = get_dataset_list(dataset_folder)
print(imglist)
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
assert os.path.exists(COCO_MODEL_PATH)==True, COCO_MODEL_PATH + '下未找到模型文件'
ImageNet_MODEL_PATH = os.path.join(ROOT_DIR, "resnet50_weights.h5")
#utils.download_trained_weights(COCO_MODEL_PATH)
MODEL_SAVE_DIR = os.path.join(ROOT_DIR, "SavedModels")


# In[3]:

iter_num = 0

class RoomDataset(utils.Dataset):
    #类别列表
    class_names = {"person"}
    #class_names = {"person","laptop","pen","bottle"}
    #从类外获取类别数量
    @classmethod
    def get_class_count(cls):
        return len(RoomDataset.class_names)
    #保存已计算的遮罩数组与标签数组
    __masks = {}

    def __init__(self):
        super(RoomDataset,self).__init__()

    #得到该图中有多少个实例（物体）
    def get_obj_index(self, maskimage):
        n = np.max(maskimage)
        return n

    #解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self,image_id):
        info = self.image_info[image_id]
        print(info['yaml_path'])
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels
                
    #重新写load_shapes，里面包含自己的类别class_list，
    #并在self.image_info信息中添加了path、mask_path、yaml_path
    def load_shapes(self, dataset_folder, imglist):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        """
        print('begin loading shapes...')
        # Add classes
        for i,class_name in enumerate(self.class_names):
            self.add_class("shapes",i + 1, class_name)
        count = len(imglist)
        for i in range(count):
            print('loading shapes of file %s' % imglist[i])
            filename = imglist[i].split(".")[0]
            rgb_path = os.path.join(dataset_folder, filename + "/img.png")            
            if not os.path.exists(rgb_path):
                raise IOError('%s not exist!' % rgb_path)
            mask_path = os.path.join(dataset_folder, filename + "/label.png")
            if not os.path.exists(mask_path):
                raise IOError('%s not exist!' % mask_path)
            yaml_path = os.path.join(dataset_folder, filename + "/info.yaml")
            if not os.path.exists(yaml_path):
                raise IOError('%s not exist!' % yaml_path)
            img = Image.open(rgb_path)
            self.add_image("shapes", image_id=i, path=rgb_path,
                           width=img.width, height=img.height, 
                           mask_path=mask_path, yaml_path=yaml_path)
            img.close()
    
    #重写load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        #根据灰度遮罩图生成mask数组
        def draw_mask(num_obj, mask, maskimage):
            info = self.image_info[image_id]
            #print('%d objects in %d * %d image.' % (num_obj,info['width'],info['height']))
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = maskimage.getpixel((i,j))  #检索指定坐标点的像素的灰度值
                    if at_pixel > 0:
                        mask[j, i, at_pixel - 1] = 1
            return mask

        if not image_id in self.__masks:
            print('begin loading masks with image ID: %d' % image_id)
            global iter_num
            info = self.image_info[image_id]
            count = 1  # number of object
            maskimg = Image.open(info['mask_path'])
            num_obj = self.get_obj_index(maskimg)
            mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
            mask = draw_mask(num_obj, mask, maskimg)
            maskimg.close() #记得关闭文件
            occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
            for i in range(count - 2, -1, -1):
                mask[:, :, i] = mask[:, :, i] * occlusion
                occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
            labels = []
            labels = self.from_yaml_get_class(image_id)
            labels_form = []
            class_counts = {}
            for i in range(len(labels)):
                for class_name in self.class_names:
                    if labels[i].find(class_name) != -1:
                        if class_name in class_counts:
                            class_counts[class_name]+=1
                        else:
                            class_counts[class_name] = 1
                        labels_form.append(class_name)
            print('classes: %s' % class_counts)
            class_ids = np.array([self.class_names.index(s) for s in labels_form])
            #将新数组保存
            self.__masks[image_id] = [mask.astype(np.bool), class_ids.astype(np.int32)]
        return self.__masks[image_id]
    
# In[4]:

class PersonsConfig(Config):
    # Give the configuration a recognizable name
    NAME = "persons"

    # Train on GPU_COUNT GPU and IMAGES_PER_GPU images per GPU.  Batch size is
    # (GPUs * images/GPU).
    GPU_COUNT = 3
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    #NUM_CLASSES = 1 + RoomDataset.get_class_count()  # background + count of shapes
    NUM_CLASSES = 1 + 1  # background + count of shapes

    # Use small images for faster training.  Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 720
    IMAGE_MAX_DIM = 1280
    IMAGE_RESIZE_MODE = "square"
    
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    # Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200
    
    # 虽然图片很大，但图中物体较小，因此使用较小的候选区
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 200

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50
    
config = PersonsConfig()
config.display()

# In[5]:
#train与val数据集准备
dataset_train = RoomDataset()
dataset_train.load_shapes(dataset_folder, imglist)
dataset_train.prepare()

dataset_val = RoomDataset()
dataset_val.load_shapes(dataset_folder, imglist)
dataset_val.prepare()

# In[6]:
# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, min(len(dataset_train.image_ids),2))
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# In[7]:
print("Prepare for training...")
    
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last
if init_with == "imagenet":
    model.load_weights(ImageNet_MODEL_PATH, by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

# In[8]:
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers.  You can also pass a regular expression to select
# which layers to train by name pattern.
print("==========First Step===========")
print("====Train the head branches====")
print("=============Begin=============")
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')
print("======First Step Finish!=======")

# In[9]:
# Fine tune all layers
# Passing layers="all" trains all layers.  You can also
# pass a regular expression to select which layers to
# train by name pattern.
print("==========Second Step==========")
print("=====Fine tune all layers======")
print("=============Begin=============")
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2, 
            layers="all")
print("======Second Step Finish!======")

# In[10]:
# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# 以下代码可以将所得模型保存至model_path
if not os.path.exists(MODEL_SAVE_DIR):
    os.mkdir(MODEL_SAVE_DIR)
model_save_name = "mask_rcnn-" + time.strftime('%y%m%d%H%M') + ".h5"
model_path = os.path.join(MODEL_SAVE_DIR, model_save_name)
model.keras_model.save_weights(model_path)
print("Model was saved into %s" % model_path)