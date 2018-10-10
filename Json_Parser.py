# -*- coding: utf-8 -*-
# @Time    : 2018/10/10/010 13:48 下午
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : Json_Parser.py
# @Software: PyCharm

import os
import json
import numpy as np
import PIL.Image
import PIL.ImageDraw
import base64
import io
import matplotlib.pyplot as plt

# 根目录
ROOT_DIR = os.path.abspath(".")
dataset_folder = os.path.join(ROOT_DIR, "dataset")  # 训练数据目录


def _get_json_files(rootdir):
    list = []
    for root, dirs, files in os.walk(rootdir):
        for name in files:
            if os.path.splitext(name)[1].lower() == '.json':
                json_path = os.path.join(root, name)
                list.append(json_path)
    return list


def _read_json(json_path):
    # json串是一个字符串
    f = open(json_path, encoding='utf-8')
    res = f.read()
    f.close()
    json_str = json.loads(res)  # 把json串，变成python的数据类型，只能转换json串内容
    return json_str


def _polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def _parse_shapes(shapes, img_shape):
    '''
    parse shapes into mask_images and keypoint_dictionarys
    :param shapes:a list of shape dictionaries
    :param img_shape: shape of the mask image, [height, width] or [height, width, channels]
    :return:
    mask_img: a numpy.ndarray with shape [height, width, channels] image
    keypoints: a 3-dim numpy.ndarray with shape [width-axis, height-axis, label_index]
    '''
    polygons_dict = {'labels': [], 'points': []}
    keypoints = []  # element is [width-axis, height-axis, label_index]
    indices_of_labelnames = {'_background_': 0}
    mask_img = np.zeros(img_shape[:2], dtype=np.int32)

    for shape in shapes:
        label_name = shape['label']
        if label_name in indices_of_labelnames:
            label_value = indices_of_labelnames[label_name]
        else:
            label_value = len(indices_of_labelnames)
        indices_of_labelnames[label_name] = label_value
        fill_color = shape['fill_color']
        line_color = shape['line_color']
        points = shape['points']
        if len(points) > 1:  # polygons
            polygons_dict['labels'].append(label_name)
            polygons_dict['points'].extend(points)
            mask = _polygons_to_mask(img_shape[:2], points)
            mask_img[mask] = indices_of_labelnames[label_name]
        else:  # keypoints
            keypoints.append([points[0][0], points[0][1], indices_of_labelnames[label_name]])
    return mask_img, np.array(keypoints)


def _img_b64_to_arr(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(PIL.Image.open(f))
    return img_arr


def _parse_json_string(json_str):
    '''
    flags = json_str['flags']
    lineColor = json_str['lineColor']
    fillColor = json_str['fillColor']
    imagePath = json_str['imagePath']
    '''
    shapes = json_str['shapes']
    imageData = json_str['imageData']
    img = _img_b64_to_arr(imageData)
    mask_img, keypoints_dict = _parse_shapes(shapes, img.shape)
    return img, mask_img, keypoints_dict


def parse_jsons(jsons_folder):
    filepaths = _get_json_files(jsons_folder)
    list = []
    for fp in filepaths:
        json_str = _read_json(fp)
        list.append(_parse_json_string(json_str))
    return list


def main():
    list = parse_jsons(dataset_folder)
    count = len(list)
    for i in range(min(5, count)):
        img, mask, _ = list[i]
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(mask)
        plt.show()


if __name__ == '__main__':
    main()
