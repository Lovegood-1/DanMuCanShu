import cv2
import numpy as np


def _polygon_fitting_for_single_object(contour):
    return np.squeeze(np.array(contour)).tolist()


def _polygon_fitting_all(contours):
    _dict = {"shapes": []}
    for i in range(len(contours)):
        single_annotation = {"label": str(
            i), "points": _polygon_fitting_for_single_object(contours[i])}
        _dict["shapes"].append(single_annotation)

    return _dict


def polygon_fitting(img_path, annot_dict):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(
        img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    shapes = _polygon_fitting_all(contours)
    annot_dict.update(shapes)
    return annot_dict


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

if __name__ == "__main__":
    img_path = r'D:\EX_AutoLabeling\example_labelme\cloud_circle.png'
    create_json(img_path)
# points= [
#         [
#           409.6666666666667,
#           333.0
#         ],
#         [
#           359.0,
#           442.33333333333337
#         ],
#         [
#           811.0,
#           457.0
#         ],
#         [
#           696.3333333333334,
#           241.0
#         ]
#       ]

# point = np.array(points)
