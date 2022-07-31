import json
import json
import numpy as np

import cv2

import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
import io

import base64

from polyfitting import *


def img_arr_to_b64(img_arr):
    img_pil = PIL.Image.fromarray(img_arr)
    f = io.BytesIO()
    # img_pil.save(f, format="PNG")
    img_pil.save(f, format="JPEG")

    img_bin = f.getvalue()
    if hasattr(base64, "encodebytes"):
        img_b64 = base64.encodebytes(img_bin)
    else:
        img_b64 = base64.encodestring(img_bin)
    # imagedata=str(base64.b64encode(img_bin), 'utf-8')
    imagedata = str(base64.b64encode(img_bin).decode("utf-8"))
    return imagedata


def generate_json(picture_path):
    """_summary_

    Args:
        picture_path (_type_): _description_
    """
    # 读取模板
    template_path = r"example_labelme/circle.json"
    with open(template_path, 'r') as load_f:
        dict_ = json.load(load_f)
    # 修改内容
    new_dict_ = dict_.copy()
    # 1 标注为空
    new_dict_['shapes'] = []
    # 2 修改其他的
    new_dict_['imagePath'] = picture_path.split('\\')[-1]
    img_pil = PIL.Image.open(img_path)
    img_arr = np.array(img_pil)
    img_b64 = img_arr_to_b64(img_arr)
    new_dict_['imageData'] = img_b64
    new_dict_['imageHeight'] = img_arr.shape[0]
    new_dict_['imageWidth'] = img_arr.shape[1]
    json_path = picture_path.split('.')[0] + '.json'
    with open(json_path, "w") as f:
        json.dump(new_dict_, f)


def create_json(img_path):
    new_dict_ = {"version": "5.0.1", "flags": {}, "shapes": [
    ], "imagePath": None, "imageData": None,  "imageHeight": None, "imageWidth": None}
    # 1 标注为空
    new_dict_['shapes'] = []
    # 2 修改其他的
    new_dict_['imagePath'] = img_path.split('\\')[-1]
    img_pil = PIL.Image.open(img_path)
    img_arr = np.array(img_pil)
    img_b64 = img_arr_to_b64(img_arr)
    new_dict_['imageData'] = img_b64
    new_dict_['imageHeight'] = img_arr.shape[0]
    new_dict_['imageWidth'] = img_arr.shape[1]
    json_path = img_path.split('.')[0] + '.json'

    new_dict_ = polygon_fitting(img_path, new_dict_)

    with open(json_path, "w") as f:
        json.dump(new_dict_, f)


img_path = r'D:\EX_AutoLabeling\example_labelme\cloud_circle.png'
create_json(img_path)
# generate_json(img_path)


img_path = r'example_labelme\1_89.jpg'


img_pil = PIL.Image.open(img_path)
img_arr = np.array(img_pil)

# img_arr = img_data_to_arr(img)
img_b64 = img_arr_to_b64(img_arr)
a = 1
