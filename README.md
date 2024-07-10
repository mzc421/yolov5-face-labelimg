<h1 align="center">💡项目结构</h1>

```
.
├── face-labeling							# 项目名称
├── ├── yolov5                                                          # yolov5 项目
│   ├── tools								# 工具包
│   │   ├── coco_json.py					        # MS COCO JSON
│   │   ├── log.py							# 日志管理
│   │   ├── obj_opt.py						        # 目标管理
│   │   └── time_format.py					        # 日期格式化
│   │   ├── voc_xml.py						        # PASCAL VOC XML
│   │   ├── yolo_txt.py						        # YOLO TXT
│   ├── data								# 测试数据
│   │   └── imgs							# 测试图片
│   │   └── videos							# 测试视频
│   ├── weights								# 推理模型模型
│   │   ├── *.pt							# PyTorch模型
│   ├── face_labeling.py					        # 主运行文件
│   ├── Face-README.md					                # 项目说明
│   └── requirements.txt					        # 脚本依赖包
```

### ✅ 基于YOLOv5的人脸检测模型的构建

📌 **widerface-m人脸检测模型**是在[WIDER FACE](http://shuoyang1213.me/WIDERFACE/)数据集上，基于[YOLOv5 v6.1](https://github.com/ultralytics/yolov5)训练的。

📌 **darkface-m人脸检测模型**是在[DARK FACE](https://flyywh.github.io/CVPRW2019LowLight/)数据集上，基于[YOLOv5 v6.1](https://github.com/ultralytics/yolov5)训练的。

<h1 align="center">⚡使用教程</h1>

### 💡 webcam实时标注

```shell
# a键捕获视频帧，q键退出
python face_labeling.py --mode webcam
```

### 💡 图片标注（包括批量图片标注）

```shell
python face_labeling.py --mode img --img_dir ./data/imgs
```

### 💡 视频标注

```shell
python face_labeling.py --mode video --img_dir ./data/videos
```

❗ 注：本项目支持的图片输入格式：**jpg** |  **jpeg** | **png** | **bmp** | **tif** | **webp**

❗ 注：本项目支持的视频输入格式：**mp4** | **avi** | **wmv** | **mkv** | **mov** | **gif** | **vob** | **swf** | **mpg** | **flv** | **3gp** | **3g2**

❗ 说明：以上三种检测模式都会在项目根目录中生成`runs`目录，该目录会生成`exp*`的子目录，子目录结构如下：

```
# webcam和图片标注的目录
.
├── runs						# 人脸数据保存目录
│   ├── exp						# 子目录
│   │   ├── raw						# 原始图片
│   │   ├── tag						# 标记图片（包括：人脸检测框、人脸ID、置信度、帧ID、FPS、人脸总数，人脸尺寸类型（小、中、大）数量）
│   │   ├── voc_xml					# PASCAL VOC XML 标注文件
│   │   ├── coco_json				# MS COCO JSON 标注文件
│   │   ├── yolo_txt				# YOLO TXT 标注文件
│   ├── frame2						# 子目录
│   │   ├── raw						# 原始图片
│   │   ├── ......
```

```
# 视频标注的目录
.
├── runs						# 人脸数据保存目录
│   ├── exp						# 子目录
│	│   ├── video_name01			# 子视频目录
│   │   │   ├── raw					# 原始图片
│   │   │   ├── tag					# 标记图片（包括：人脸检测框、人脸ID、置信度、帧ID、FPS、人脸总数，人脸尺寸类型（小、中、大）数量）
│   │   │   ├── voc_xml				# PASCAL VOC XML 标注文件
│   │   │   ├── coco_json			# MS COCO JSON 标注文件
│   │   │   ├── yolo_txt			# YOLO TXT 标注文件
│	│   ├── video_name02			# 子视频目录
│   │   │   ├── raw					# 原始图片
│   │   │   ├── ......
```

查看检测结果：人脸图片检测结果会保存在`runs/exp*/tag`中。
