# YOLO TXT导出
# 创建人：曾逸夫
# 创建时间：2022-03-18

import sys

ROOT_PATH = sys.path[0]


# yolo txt
def xyxy2yolo(img_shape, xyxy_list):

    x0, y0, x1, y1 = xyxy_list
    xc = (x0 + x1) // 2  # 中心点x center
    yc = (y0 + y1) // 2  # 中心点y center

    bw = x1 - x0  # 边框宽度
    bh = y1 - y0  # 边框高度

    imgH, imgW, _ = img_shape  # 图片尺寸

    # -------yolo 坐标格式化-------
    yolo_0 = round((xc / imgW), 6)
    yolo_1 = round((yc / imgH), 6)
    yolo_2 = round((bw / imgW), 6)
    yolo_3 = round((bh / imgW), 6)

    return yolo_0, yolo_1, yolo_2, yolo_3


# 创建yolo txt
def create_yolo_txt(clsName_list, img_shape, xyxy_list, output_txt):
    f = open(f"{output_txt}", "w")

    for i in range(len(xyxy_list)):
        # 格式化坐标
        yolo_0, yolo_1, yolo_2, yolo_3 = xyxy2yolo(img_shape, xyxy_list[i])
        cls_name = clsName_list[i][0]  # 类别索引
        f.write(f"{cls_name},{yolo_0},{yolo_1},{yolo_2},{yolo_3}\n")

    f.close()
