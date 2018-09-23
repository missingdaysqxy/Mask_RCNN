from PIL import Image
import numpy as np
import math
import os

dataset_root_path = os.path.abspath(".")    #训练数据根目录
input_json = os.path.join(dataset_root_path, "json")    #将labelme_json_to_dataset.exe生成的文件夹存入此处
output_mask = os.path.join(dataset_root_path, "mask")   #灰度遮罩图输出目录
output_rgb = os.path.join(dataset_root_path, "rgb")     #原始彩色图像输出目录

def toeight():
    list = os.listdir(input_json)  # 该文件夹下所有的文件与子文件夹名称
    for name in list:
        print("process %s" % name)
        #处理label.png
        input_label = os.path.join(input_json, name + '/label.png')
        if not os.path.exists(input_label):
            continue
        img_label = Image.open(input_label)  # 打开图片
        img_label = np.array(img_label)
        img_label = Image.fromarray(np.uint8(img_label))
        img_label.save(os.path.join(output_mask, name + '.png'))
        #处理img.png
        input_rgb = os.path.join(input_json, name + '/img.png')
        if not os.path.exists(input_rgb):
            continue
        img_rgb = Image.open(input_rgb)
        img_rgb.save(os.path.join(output_rgb, name + '.png'))
    print("finish!")
toeight()