# Pascal VOC XML导出
# 创建人：曾逸夫
# 创建时间：2022-03-12

from xml.dom.minidom import Document


# 创建xml节点
def create_xmlNode(doc, up_node, new_node_name):
    new_node = doc.createElement(new_node_name)
    up_node.appendChild(new_node)

    return new_node


# 创建节点内容
def create_xmlNodeContent(doc, new_node, new_node_content):
    new_nodeContent = doc.createTextNode(new_node_content)
    new_node.appendChild(new_nodeContent)


# 创建xml文件
def create_xml(
    imgName,
    imgPath,
    img_shape,
    clsName_list,
    xyxy_list,
    obj_size_style_list,
    output_xml,
):
    doc = Document()  # 实例化xml文件对象

    annotation = doc.createElement("annotation")
    doc.appendChild(annotation)

    folder = create_xmlNode(doc, annotation, "folder")
    create_xmlNodeContent(doc, folder, "")
    filename = create_xmlNode(doc, annotation, "filename")  # 图片名称
    create_xmlNodeContent(doc, filename, imgName)

    path = create_xmlNode(doc, annotation, "path")  # 图片路径
    create_xmlNodeContent(doc, path, imgPath)

    source = create_xmlNode(doc, annotation, "source")
    database = create_xmlNode(doc, source, "database")
    create_xmlNodeContent(doc, database, "FaceDatabase")  # 数据集名称

    # ------------图片尺寸------------
    size = create_xmlNode(doc, annotation, "size")
    width = create_xmlNode(doc, size, "width")
    create_xmlNodeContent(doc, width, str(img_shape[1]))
    height = create_xmlNode(doc, size, "height")
    create_xmlNodeContent(doc, height, str(img_shape[0]))
    depth = create_xmlNode(doc, size, "depth")
    create_xmlNodeContent(doc, depth, str(img_shape[2]))

    segmented = create_xmlNode(doc, annotation, "segmented")
    create_xmlNodeContent(doc, segmented, "0")

    obj_num = create_xmlNode(doc, annotation, "obj_num")  # 实例数量
    create_xmlNodeContent(doc, obj_num, str(len(xyxy_list)))

    for i in range(len(xyxy_list)):
        # ------------Object------------
        object = create_xmlNode(doc, annotation, "object")
        name = create_xmlNode(doc, object, "name")  # 类别名称
        create_xmlNodeContent(doc, name, clsName_list[i][1])
        pose = create_xmlNode(doc, object, "pose")
        create_xmlNodeContent(doc, pose, "Unspecified")
        truncated = create_xmlNode(doc, object, "truncated")
        create_xmlNodeContent(doc, truncated, "0")
        difficult = create_xmlNode(doc, object, "difficult")
        create_xmlNodeContent(doc, difficult, "0")
        occluded = create_xmlNode(doc, object, "occluded")
        create_xmlNodeContent(doc, occluded, "0")

        obj_size = create_xmlNode(doc, object, "obj_size")  # 实例大小，小目标,中目标,大目标
        create_xmlNodeContent(doc, obj_size, obj_size_style_list[i])

        # ------------边框坐标------------
        bndbox = create_xmlNode(doc, object, "bndbox")
        xmin = create_xmlNode(doc, bndbox, "xmin")
        create_xmlNodeContent(doc, xmin, str(xyxy_list[i][0]))
        xmax = create_xmlNode(doc, bndbox, "xmax")
        create_xmlNodeContent(doc, xmax, str(xyxy_list[i][2]))
        ymin = create_xmlNode(doc, bndbox, "ymin")
        create_xmlNodeContent(doc, ymin, str(xyxy_list[i][1]))
        ymax = create_xmlNode(doc, bndbox, "ymax")
        create_xmlNodeContent(doc, ymax, str(xyxy_list[i][3]))

    f = open(output_xml, "w")
    f.write(doc.toprettyxml(indent="  "))
    f.close()
