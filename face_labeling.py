# Face Labeling v0.2.2
# 创建人：曾逸夫
# 创建时间：2022-07-20

import argparse
import gc
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from tools.coco_json import coco_json_main
from tools.log import rich_log
from tools.obj_opt import get_obj_size
from tools.time_format import time_format
from tools.voc_xml import create_xml
from tools.yolo_txt import create_yolo_txt

from models.common import DetectMultiBackend
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox
from utils.general import non_max_suppression, Profile, increment_path, scale_boxes

# ---------------------图片和视频输入格式---------------------
IMG_FORMATS = ["jpg", "jpeg", "png", "bmp", "tif", "webp"]
VIDEO_FORMATS = ["mp4", "avi", "wmv", "mkv", "mov", "gif", "vob", "swf", "mpg", "flv", "3gp", "3g2"]

ROOT_PATH = sys.path[0]  # 根目录
FACELABELING_VERISON = "Face Labeling v1.0"

coco_imgs_list = []  # 图片列表（COCO）
coco_anno_list = []  # 标注列表（COCO）
categories_id = 0  # 类别ID（COCO）

color_list = [(0, 0, 255), (0, 255, 0), (181, 228, 255)]

console = Console()


def parse_args(known=False):
    parser = argparse.ArgumentParser(description="Face Labeling v0.2.2")
    parser.add_argument("--device", default="0", type=str, help="cuda or cpu")
    parser.add_argument("--mode", default="video", type=str, choices=['video', 'img', 'webcam'], help="face labeling mode")
    parser.add_argument("--img_dir",
                        default=r"D:\BaiduNetdiskDownload\face-labeling-master\face-labeling-master\data\imgs",
                        type=str, help="image dir")
    parser.add_argument("--video_dir",
                        default=r"D:\BaiduNetdiskDownload\face-labeling-master\face-labeling-master\data\videos",
                        type=str, help="video dir")
    parser.add_argument("--model_name", default="widerface-m.pt", type=str, help="model name")
    parser.add_argument("--imgName", default="face_test", type=str, help="image name")
    parser.add_argument("--project", default="runs", type=str, help="frame save dir")
    parser.add_argument("--exp", default="exp", type=str, help="frame dir name")
    parser.add_argument("--nms_conf", default=0.45, type=float, help="model NMS confidence threshold")
    parser.add_argument("--nms_iou", default=0.25, type=float, help="model NMS IoU threshold")
    parser.add_argument("--max_det", default=1000, type=int, help="model max detect obj num")
    parser.add_argument("--img_size", default=640, type=int, help="model inference size")
    parser.add_argument("--label_no_show", action="store_true", default=False, help="label show")
    parser.add_argument("--label_simple", default="dnt", type=str, choices=["dnt", "id", "conf"], help="label simple, dnt or id or conf")
    parser.add_argument("--label_progressBar", default="bar", type=str, choices=["bar", "dnt", "bar"], help="label progress bar, dnt or bar")
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


# 人脸检测与信息提取
def face_detect(file_path, mode, frame, model, frame_id, face_id, stride, names, auto, max_det, frame_savePath, conf_thres,
                iou_thres, imgName, img_size=640, label_no_show=False, label_simple="dnt",
                label_progressBar="dnt", video_name="vide_name.mp4"):
    global coco_imgs_list, coco_anno_list, categories_id

    wait_key = cv2.waitKey(20) & 0xFF  # 键盘监听
    xyxy_list = []  # xyxy 列表置空
    obj_size_style_list = []  # 目标尺寸类型
    clsName_list = []  # 类别列表

    img_shape = frame.shape  # 帧尺寸
    frame_cp = frame.copy()  # 原始图片副本
    im = letterbox(frame, img_size, stride=stride, auto=auto)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    dt = Profile(device=model.device)
    with dt:
        pred = model(im)

    pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

    # 显示帧ID
    cv2.putText(frame, f"Frame ID: {frame_id}", (0, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)

    color_draw = color_list[1]

    for id, det in enumerate(pred):
        fps = f"{(1000 * float(dt.t)):.2f}"  # FPS
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # 类别索引
                label = names[c]
                clsName_list.append([c, label])

                x0, y0, x1, y1 = [int(i.cpu().tolist()) for i in xyxy]
                xyxy_list.append([x0, y0, x1, y1])  # 边框坐标列表
                obj_size = get_obj_size([x0, y0, x1, y1])  # 获取目标尺寸

                # --------标签和边框颜色设置--------
                if obj_size == "small":
                    color_draw = color_list[0]
                elif obj_size == "medium":
                    color_draw = color_list[1]
                elif obj_size == "large":
                    color_draw = color_list[2]

                obj_size_style_list.append(obj_size)  # 获取目标尺寸列表

                conf = float(conf)  # 置信度

                if not label_no_show:
                    # --------标签样式--------
                    if label_simple == "dnt":
                        label_style = f"{id}-{label}:{conf:.2f}"
                    elif label_simple == "id":
                        label_style = f"{id}"
                    elif label_simple == "conf":
                        label_style = f"{conf * 100:.0f}%"

                    # 标签背景尺寸
                    labelbg_size = cv2.getTextSize(label_style, cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)

                    # 标签背景
                    if label_progressBar == "dnt":
                        cv2.rectangle(frame, (x0, y0), (x0 + labelbg_size[0][0], y0 + labelbg_size[0][1]),
                                      color_draw, thickness=-1)
                    elif label_progressBar == "bar":
                        cv2.rectangle(frame, (x0, y0), (x0 + int((x1 - x0) * conf), y0 + labelbg_size[0][1]),
                                      color_draw, thickness=-1)

                # 标签
                cv2.putText(frame, label_style, (x0, y0 + labelbg_size[0][1]), cv2.FONT_HERSHEY_COMPLEX,
                            0.6, (0, 0, 0), 1)

                # 检测框
                cv2.rectangle(frame, (x0, y0), (x1, y1), color_draw, 2)

            # 变量回收
            del id, c, label, x0, y0, x1, y1, conf

        # FPS
        cv2.putText(frame, f"FPS: {fps}", (0, 20), cv2.FONT_HERSHEY_COMPLEX,
                    0.6, (0, 255, 0), 1)

    print(f"{file_path} {pred[0].shape[0] if len(det) else 'no'} detections, {dt.t * 1E3:.1f}ms")

    # 人脸数量
    cv2.putText(frame, f"Face Num: {len(xyxy_list)}", (0, 60), cv2.FONT_HERSHEY_COMPLEX,
                0.6, (0, 255, 0), 1)

    # ---------------------目标尺寸类型---------------------
    small_num = Counter(obj_size_style_list)["small"]  # 小目标
    medium_num = Counter(obj_size_style_list)["medium"]  # 中目标
    large_num = Counter(obj_size_style_list)["large"]  # 大目标

    cv2.putText(frame, f"small: {small_num}", (0, 80), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, f"medium: {medium_num}", (0, 100), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, f"large: {large_num}", (0, 120), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)

    imgName_faceid = f"{imgName}-{face_id}"  # 图片名称-FaceID

    if mode == "webcam" and wait_key == ord("a"):
        # 捕获视频帧
        cv2.imwrite(f"{frame_savePath}/org/{imgName_faceid}.jpg", frame_cp)  # 保存原始图片
        cv2.imwrite(f"{frame_savePath}/tag/{imgName_faceid}.jpg", frame)  # 保存标记图片

        # 创建VOC XML文件
        create_xml(f"{imgName_faceid}.jpg", f"{frame_savePath}/voc_xml/{imgName_faceid}.jpg",
                   img_shape, clsName_list, xyxy_list, obj_size_style_list,
                   f"{frame_savePath}/voc_xml/{imgName_faceid}.xml")

        create_yolo_txt(clsName_list, img_shape, xyxy_list, f"{frame_savePath}/yolo_txt/{imgName_faceid}.txt")

        # ------------加入coco图片信息和标注信息------------
        coco_imgs_list.append([face_id, f"{imgName_faceid}.jpg", img_shape[1], img_shape[0],
                               f"{datetime.now():%Y-%m-%d %H:%M:%S}", ])
        coco_anno_list.append(
            [[categories_id + i for i in range(len(xyxy_list))], face_id, clsName_list, xyxy_list])
        categories_id += len(xyxy_list)

        face_id += 1  # 人脸ID自增

    elif mode == "img":
        # 捕获视频帧
        cv2.imwrite(f"{frame_savePath}/raw/{imgName_faceid}.jpg", frame_cp)  # 保存原始图片
        cv2.imwrite(f"{frame_savePath}/tag/{imgName_faceid}.jpg", frame)  # 保存标记图片

        # 创建VOC XML文件
        create_xml(f"{imgName_faceid}.jpg", f"{frame_savePath}/voc_xml/{imgName_faceid}.jpg",
                   img_shape, clsName_list, xyxy_list, obj_size_style_list,
                   f"{frame_savePath}/voc_xml/{imgName_faceid}.xml", )

        create_yolo_txt(clsName_list, img_shape, xyxy_list, f"{frame_savePath}/yolo_txt/{imgName_faceid}.txt")

        # ------------加入coco图片信息和标注信息------------
        coco_imgs_list.append([face_id,
                               f"{imgName_faceid}.jpg", img_shape[1], img_shape[0],
                               f"{datetime.now():%Y-%m-%d %H:%M:%S}", ])
        coco_anno_list.append([[categories_id + i for i in range(len(xyxy_list))],
                               face_id, clsName_list, xyxy_list])
        categories_id += len(xyxy_list)

        face_id += 1  # 人脸ID自增

    elif mode == "video":
        # 捕获视频帧
        cv2.imwrite(f"{frame_savePath}/{video_name}/raw/{imgName_faceid}.jpg", frame_cp)  # 保存原始图片
        cv2.imwrite(f"{frame_savePath}/{video_name}/tag/{imgName_faceid}.jpg", frame)  # 保存标记图片

        # 创建VOC XML文件
        create_xml(f"{imgName_faceid}.jpg", f"{frame_savePath}/{video_name}/voc_xml/{imgName_faceid}.jpg",
                   img_shape, clsName_list, xyxy_list, obj_size_style_list,
                   f"{frame_savePath}/{video_name}/voc_xml/{imgName_faceid}.xml")

        create_yolo_txt(clsName_list, img_shape, xyxy_list,
                        f"{frame_savePath}/{video_name}/yolo_txt/{imgName_faceid}.txt")

        # ------------加入coco图片信息和标注信息------------
        coco_imgs_list.append([face_id, f"{imgName_faceid}.jpg", img_shape[1], img_shape[0],
                               f"{datetime.now():%Y-%m-%d %H:%M:%S}", ])
        coco_anno_list.append([[categories_id + i for i in range(len(xyxy_list))], face_id, clsName_list, xyxy_list])

        categories_id += len(xyxy_list)

        face_id += 1  # 人脸ID自增

    if mode == "webcam":
        return frame, wait_key, face_id
    elif mode == "img":
        return frame, face_id
    elif mode == "video":
        return frame, face_id


@smart_inference_mode()
def face_label(device="0", mode="webcam", img_dir="./data/imgs", video_dir="./data/videos", model_name="widerface-m",
               imgName="face_test", project="runs", exp="exp", nms_conf=0.25, nms_iou=0.45, max_det=1000, img_size=640,
               label_no_show=False, label_simple="dnt", label_progressBar="dnt"):
    device = select_device(device)
    model = DetectMultiBackend(f"{ROOT_PATH}/weights/{model_name}", device=device)
    stride, names, pt = model.stride, model.names, model.pt

    # ----------创建帧文件----------
    frame_savePath = increment_path(Path(f"{ROOT_PATH}/{project}") / exp, exist_ok=False)  # 增量运行

    frame_savePath.mkdir(parents=True, exist_ok=True)  # 创建目录

    if mode in ["webcam", "img"]:
        # 创建原始图片目录
        Path(f"{frame_savePath}/raw").mkdir(parents=True, exist_ok=True)
        # 创建标记图片目录
        Path(f"{frame_savePath}/tag").mkdir(parents=True, exist_ok=True)
        # 创建PASCAL VOC XML目录
        Path(f"{frame_savePath}/voc_xml").mkdir(parents=True, exist_ok=True)
        # 创建MS COCO JSON目录
        Path(f"{frame_savePath}/coco_json").mkdir(parents=True, exist_ok=True)
        # 创建YOLO TXT目录
        Path(f"{frame_savePath}/yolo_txt").mkdir(parents=True, exist_ok=True)

    face_id = 0  # 人脸ID
    frame_id = 0  # 帧ID

    logTime = f"{datetime.now():%Y-%m-%d %H:%M:%S}"  # 日志时间
    rich_log(f"{logTime}\n")  # 记录日志时间

    s_time = time.time()  # 起始时间
    console.rule(f"🔥 {FACELABELING_VERISON} 程序开始！")

    if mode == "webcam":
        cap = cv2.VideoCapture(0)  # 连接设备
        is_capOpened = cap.isOpened()  # 判断设备是否开启
        count = 0
        frame_width = int(cap.get(3))  # 帧宽度
        frame_height = int(cap.get(4))  # 帧高度
        fps = cap.get(5)  # 帧率
        # 调用face webcam
        if is_capOpened:
            vid_writer = cv2.VideoWriter(f"{frame_savePath}//tag/webcam.mp4",
                                         cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                         (frame_width, frame_height))

            print(f"🚀 欢迎使用{FACELABELING_VERISON}，摄像头连接成功！\n")  # 摄像头连接成功提示
            while is_capOpened:
                _, frame = cap.read()  # 帧读取
                cv2.namedWindow(FACELABELING_VERISON)  # 设置窗口
                count += 1
                file_path = count
                # 人脸检测与信息提取
                frame, wait_key, face_id = face_detect(file_path, mode, frame, model, frame_id, face_id, stride, names,
                                                       pt, max_det, frame_savePath, nms_conf, nms_iou,
                                                       imgName, img_size, label_no_show, label_simple,
                                                       label_progressBar)

                vid_writer.write(frame)
                cv2.imshow(FACELABELING_VERISON, frame)  # 显示

                if wait_key == ord("q"):
                    # 退出窗体
                    break

                frame_id += 1  # 帧ID自增

                # 变量回收
                del frame
                gc.collect()

            vid_writer.release()
            cap.release()
            coco_json_main(names, coco_imgs_list, coco_anno_list, f"{frame_savePath}/coco_json/face_coco.json")

        else:
            print("摄像头连接异常！")

    elif mode == "img":
        # 筛选图片文件
        imgName_list = [i for i in os.listdir(img_dir) if i.split(".")[-1].lower() in IMG_FORMATS]
        # 调用 face images
        for i in imgName_list:
            file_path = f"{img_dir}/{i}"
            frame = cv2.imread(file_path)

            frame, face_id = face_detect(file_path, mode, frame, model, frame_id, face_id, stride, names, pt, max_det,
                                         frame_savePath, nms_conf, nms_iou, imgName, img_size,
                                         label_no_show, label_simple, label_progressBar)

            frame_id += 1  # 帧ID自增

            # 变量回收
            del frame
            gc.collect()

        coco_json_main(names, coco_imgs_list, coco_anno_list, f"{frame_savePath}/coco_json/face_coco.json")

    elif mode == "video":
        # 筛选图片文件
        videoName_list = [i for i in os.listdir(video_dir) if i.split(".")[-1].lower() in VIDEO_FORMATS]

        for i in videoName_list:
            video_path = os.path.join(video_dir, i)
            input_video = cv2.VideoCapture(video_path)
            is_capOpened = input_video.isOpened()

            frame_width = int(input_video.get(3))  # 帧宽度
            frame_height = int(input_video.get(4))  # 帧高度
            fps = input_video.get(5)  # 帧率
            video_frames = int(input_video.get(7))  # 总帧数

            video_name = i.replace(".", "_")  # 点号取代下划线

            print(f"{video_name}，帧宽度：{frame_width}，帧高度：{frame_height}，帧率：{fps}，总帧数：{video_frames}")
            count = 0

            if is_capOpened:
                # 创建原始图片目录
                Path(f"{frame_savePath}/{video_name}/raw").mkdir(parents=True, exist_ok=True)
                # 创建标记图片目录
                Path(f"{frame_savePath}/{video_name}/tag").mkdir(parents=True, exist_ok=True)
                # 创建PASCAL VOC XML目录
                Path(f"{frame_savePath}/{video_name}/voc_xml").mkdir(parents=True, exist_ok=True)
                # 创建MS COCO JSON目录
                Path(f"{frame_savePath}/{video_name}/coco_json").mkdir(parents=True, exist_ok=True)
                # 创建YOLO TXT目录
                Path(f"{frame_savePath}/{video_name}/yolo_txt").mkdir(parents=True, exist_ok=True)

                vid_writer = cv2.VideoWriter(f"{frame_savePath}/{video_name}/tag/{i}",
                                             cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                             (frame_width, frame_height))

                while is_capOpened:
                    ret, frame = input_video.read()
                    if not ret:
                        # 判断空帧
                        break
                    file_path = f'{video_path} [{count}/{video_frames}]'
                    count += 1
                    frame, face_id = face_detect(file_path, mode, frame, model, frame_id, face_id, stride, names, pt,
                                                 max_det, frame_savePath, nms_conf, nms_iou,
                                                 imgName, img_size, label_no_show, label_simple,
                                                 label_progressBar, video_name)

                    vid_writer.write(frame)
                    del frame
                    gc.collect()

                coco_json_main(names, coco_imgs_list, coco_anno_list,
                               f"{frame_savePath}/{video_name}/coco_json/face_coco.json")

                input_video.release()
                cv2.destroyAllWindows()

                frame_save_msg = f"共计{face_id}张人脸图片，保存至{frame_savePath}/{video_name}/raw"
                frametag_save_msg = f"共计{face_id}张人脸标记图片，保存至{frame_savePath}/{video_name}/tag"
                xml_save_msg = f"共计{face_id}个人脸xml文件，保存至{frame_savePath}/{video_name}/voc_xml"
                json_save_msg = f"共计1个人脸json文件，保存至{frame_savePath}/{video_name}/coco_json"
                txt_save_msg = f"共计{face_id}个人脸txt文件，保存至{frame_savePath}/{video_name}/yolo_txt"
                rich_log(f"{frame_save_msg}\n{frametag_save_msg}\n{xml_save_msg}\n{json_save_msg}\n{txt_save_msg}\n")

                # rich table
                table = Table(title=f"{FACELABELING_VERISON} 保存信息", show_header=True, header_style="bold #FF6363")
                table.add_column("属性", justify="right", style="#FFAB76")
                table.add_column("个数", justify="center", style="#FFFDA2")
                table.add_column("保存路径", justify="left", style="#BAFFB4", no_wrap=True)

                table.add_row("人脸图片", f"{face_id}", f"{frame_savePath}/{video_name}/raw")
                table.add_row("人脸标记图片", f"{face_id}", f"{frame_savePath}/{video_name}/tag")
                table.add_row("人脸XML文件", f"{face_id}", f"{frame_savePath}/{video_name}/voc_xml")
                table.add_row("人脸JSON文件", "1", f"{frame_savePath}/{video_name}/coco_json")
                table.add_row("人脸TXT文件", f"{face_id}", f"{frame_savePath}/{video_name}/yolo_txt")

                console.print(table)

                face_id = 0
                vid_writer.release()
                input_video.release()

            else:
                print("连接视频失败！程序退出！")
                sys.exit()

    else:
        print("模式错误，程序退出！")
        sys.exit()

    # ------------------程序结束------------------
    console.rule(f"🔥 {FACELABELING_VERISON} 程序结束！")

    e_time = time.time()  # 终止时间
    total_time = e_time - s_time  # 程序用时

    # 格式化时间格式，便于观察
    outTimeMsg = f"用时：{time_format(total_time)}"
    print(outTimeMsg)  # 打印用时
    rich_log(f"{outTimeMsg}\n")  # 记录用时

    if mode in ["webcam", "img"]:
        frame_save_msg = f"共计{face_id}张人脸图片，保存至{frame_savePath}/raw"
        frametag_save_msg = f"共计{face_id}张人脸标记图片，保存至{frame_savePath}/tag"
        xml_save_msg = f"共计{face_id}个人脸xml文件，保存至{frame_savePath}/voc_xml"
        json_save_msg = f"共计1个人脸json文件，保存至{frame_savePath}/coco_json"
        txt_save_msg = f"共计{face_id}个人脸txt文件，保存至{frame_savePath}/yolo_txt"

        rich_log(f"{frame_save_msg}\n{frametag_save_msg}\n{xml_save_msg}\n{json_save_msg}\n{txt_save_msg}\n")

        table = Table(title=f"{FACELABELING_VERISON} 保存信息", show_header=True, header_style="bold #FF6363")
        table.add_column("属性", justify="right", style="#FFAB76")
        table.add_column("个数", justify="center", style="#FFFDA2")
        table.add_column("保存路径", justify="left", style="#BAFFB4", no_wrap=True)

        table.add_row("人脸图片", f"{face_id}", f"{frame_savePath}/raw")
        table.add_row("人脸标记图片", f"{face_id}", f"{frame_savePath}/tag")
        table.add_row("人脸XML文件", f"{face_id}", f"{frame_savePath}/voc_xml")
        table.add_row("人脸JSON文件", "1", f"{frame_savePath}/coco_json")
        table.add_row("人脸TXT文件", f"{face_id}", f"{frame_savePath}/yolo_txt")

        console.print(table)


def main(args):
    face_label(**vars(args))


if __name__ == "__main__":
    args = parse_args()
    main(args)
