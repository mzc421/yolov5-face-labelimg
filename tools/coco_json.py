# -*- coding:utf-8 -*-
# @author: 牧锦程
# @微信公众号: AI算法与电子竞赛
# @Email: m21z50c71@163.com
# @VX：fylaicai


import json
import sys

ROOT_PATH = sys.path[0]


# coco json 相关信息，可自定义
def coco_info():
    info = {
        "info": {
            "year": "",
            "version": "",
            "description": "",
            "contributor": "",
            "url": "",
            "date_created": "",}}

    return info


# coco json 许可信息，可自定义
def coco_licenses():
    licenses = {"licenses": [{"id": 1, "url": "", "name": ""}]}
    return licenses


# coco json 类别信息
# id=0, name="", supercategory:""
def coco_categories(coco_categories_list):
    coco_categories_list_cp = coco_categories_list.copy()
    coco_categories_list = []
    coco_categories_list.append([0, "facelabel", "None"])
    for i in range(len(coco_categories_list_cp)):
        coco_categories_list.append([i + 1, coco_categories_list_cp[i], "facelabel"])

    categories_list = []
    for i in range(len(coco_categories_list)):
        imgs_dict = {}
        imgs_dict["id"] = int(coco_categories_list[i][0])  # 类别ID
        imgs_dict["name"] = coco_categories_list[i][1]  # 图片名称
        imgs_dict["supercategory"] = coco_categories_list[i][2]  # 图片超类

        categories_list.append(imgs_dict)

    categories = {}
    categories["categories"] = categories_list

    return categories


# coco json 图片信息
# id=0, license=1, file_name="", height=0, width=0, date_captured=""
def coco_images(coco_imgs_list):
    coco_imgs_list_cp = coco_imgs_list.copy()
    coco_imgs_list = []
    for i in range(len(coco_imgs_list_cp)):
        coco_imgs_list.append([
            coco_imgs_list_cp[i][0],
            1,
            coco_imgs_list_cp[i][1],
            coco_imgs_list_cp[i][2],
            coco_imgs_list_cp[i][3],
            coco_imgs_list_cp[i][4],])

    imgs_list = []
    for i in range(len(coco_imgs_list)):
        imgs_dict = {}
        imgs_dict["id"] = int(coco_imgs_list[i][0])  # 图片ID
        imgs_dict["license"] = int(coco_imgs_list[i][1])  # 许可ID
        imgs_dict["file_name"] = coco_imgs_list[i][2]  # 图片名称
        imgs_dict["height"] = int(coco_imgs_list[i][3])  # 图片高度
        imgs_dict["width"] = int(coco_imgs_list[i][4])  # 图片宽度
        imgs_dict["date_captured"] = coco_imgs_list[i][5]  # 创建日期

        imgs_list.append(imgs_dict)

    images = {}
    images["images"] = imgs_list
    return images


# coco json 标注信息
# id=0, image_id=0, category_id=1, bbox=[x0, y0, bw, bh]
def coco_annotations(coco_anno_list):
    coco_anno_list_cp = coco_anno_list.copy()
    coco_anno_list = []
    for i in range(len(coco_anno_list_cp)):
        coco_anno_list.append([
            coco_anno_list_cp[i][0],
            coco_anno_list_cp[i][1],
            coco_anno_list_cp[i][2][0],
            coco_anno_list_cp[i][3],])

    annotations_list = []

    for i in range(len(coco_anno_list)):
        for j in range(len(coco_anno_list[i][0])):
            annotations_dict = {}
            annotations_dict["id"] = int(coco_anno_list[i][0][j])  # 标记ID
            annotations_dict["image_id"] = int(coco_anno_list[i][1])  # 图片ID
            annotations_dict["category_id"] = int(coco_anno_list[i][2][0]) + 1  # 类别ID

            # -----------边框信息-----------
            x0 = coco_anno_list[i][3][j][0]
            y0 = coco_anno_list[i][3][j][1]
            x1 = coco_anno_list[i][3][j][2]
            y1 = coco_anno_list[i][3][j][3]
            bw = x1 - x0
            bh = y1 - y0

            annotations_dict["bbox"] = [x0, y0, bw, bh]  # 边框坐标
            annotations_dict["area"] = bw * bh  # 边框面积
            annotations_dict["segmentation"] = []
            annotations_dict["iscrowd"] = 0

            annotations_list.append(annotations_dict)

    annotations = {}
    annotations["annotations"] = annotations_list

    return annotations


# 创建COCO JSON文件
def json_create(info, licenses, categories, images, annotations, json_path):
    coco_json = {}

    json_key = [info, licenses, categories, images, annotations]
    for i in json_key:
        coco_json.update(i)

    # json格式化保存
    with open(json_path, "w") as write_f:
        json.dump(coco_json, write_f, indent=4, ensure_ascii=False)
        write_f.close()


# COCO JSON主运行文件
def coco_json_main(coco_categories_list, coco_imgs_list, coco_anno_list, json_path):
    info = coco_info()
    licenses = coco_licenses()
    categories = coco_categories(coco_categories_list)
    images = coco_images(coco_imgs_list)
    annotations = coco_annotations(coco_anno_list)

    json_create(info, licenses, categories, images, annotations, json_path)
