# -*- coding:utf-8 -*-
# @author: 牧锦程
# @微信公众号: AI算法与电子竞赛
# @Email: m21z50c71@163.com
# @VX：fylaicai


import sys

OBJ_SIZE = [0, 32 ** 2, 96 ** 2]  # 目标尺寸阈值


def get_obj_size(xyxy_list):
    x0, y0, x1, y1 = xyxy_list  # 边框坐标
    bbox_area = (x1 - x0) * (y1 - y0)  # 边框面积

    if OBJ_SIZE[0] < bbox_area <= OBJ_SIZE[1]:
        return "small"
    elif OBJ_SIZE[1] < bbox_area < OBJ_SIZE[2]:
        return "medium"
    elif bbox_area > OBJ_SIZE[2]:
        return "large"
    else:
        print("尺寸错误，程序终止。")
        sys.exit()
