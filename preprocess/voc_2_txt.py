import os
import random
import xml.etree.ElementTree as ET
import numpy as np

'''
如果不是VOC数据集则不需要使用此脚本，可自行写数据转换脚本，或者改dataset读取形式

voc_2_txt.py用于将xml数据转换为txt格式数据
txt中的每一行数据格式为:
    [img_path, box1, box2...]
    例：G:\DeepLearning\zoomtoolkit\datasets\VOC07+12+test\VOCdevkit/VOC2007/JPEGImages/000002.jpg 139,200,207,301,18
    
使用方法：
1. 修改classes_path：此txt用于存放类别名，一行就是一个类别名
2. 修改VOCdevkit_path：数据集目录，里面的格式应该是 -VOC2007
                                                --Annotations
                                                --ImageSets
                                                --JPEGImages
3. 修改VOCdevkit_sets， 例：[('2007', 'train'), ('2007', 'val')]则生成2007_train.txt和2007_val.txt两个文件
4. 如果是使用自己的数据集，则需要将以下代码中'VOC2007'改成自己的文件夹名称                        
'''


#---------------------------------------------------#
#   从txt文件获得类，传入的是model_data/voc_classes.txt
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

# --------------------------------------------------------------------------------------------------------------------------------#
#   annotation_mode用于指定该文件运行时计算的内容
#   annotation_mode为0代表整个标签处理过程，包括获得VOCdevkit/VOC2007/ImageSets里面的txt以及训练用的2007_train.txt、2007_val.txt
#   annotation_mode为1代表获得VOCdevkit/VOC2007/ImageSets里面的txt
#   annotation_mode为2代表获得训练用的2007_train.txt、2007_val.txt
# --------------------------------------------------------------------------------------------------------------------------------#
annotation_mode = 0
# -------------------------------------------------------------------#
#   必须要修改，用于生成2007_train.txt、2007_val.txt的目标信息
#   与训练和预测所用的classes_path一致即可
#   如果生成的2007_train.txt里面没有目标信息
#   那么就是因为classes没有设定正确
#   仅在annotation_mode为0和2的时候有效
# -------------------------------------------------------------------#
classes_path = r'G:\DeepLearning\zoomtoolkit\yolov3\pytorch\model_data\voc_classes.txt'
# --------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
#   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1
#   仅在annotation_mode为0和1的时候有效
# --------------------------------------------------------------------------------------------------------------------------------#
trainval_percent = 0.9
train_percent = 0.9
# -------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
# -------------------------------------------------------#
VOCdevkit_path = r'G:\DeepLearning\zoomtoolkit\datasets\VOC07+12+test\VOCdevkit'

VOCdevkit_sets = [('2007', 'train'), ('2007', 'val')]
classes, _ = get_classes(classes_path)

# -------------------------------------------------------#
#   统计目标数量
# -------------------------------------------------------#
photo_nums = np.zeros(len(VOCdevkit_sets))
nums = np.zeros(len(classes))

'''
读取xml文件
'''
def convert_annotation(year, image_id, list_file):
    '''
    :param year: VOC数据集的年份，分为2007和2012
    :param image_id: image文件的id，不含后缀
    :param list_file: 传入一个新的txt文件用于存放转换后的数据
    :return: 直接在传入的txt上做修改，所以没有返回文件
    '''
    # 打开对应的xml文件
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml' % (year, image_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    # 遍历每一个'object'，获取其中的信息
    for obj in root.iter('object'):
        # difficult字段表示目标是否很难识别
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        # name字段是类别
        cls = obj.find('name').text
        # 如果cls不在定义好的class列表里，或者目标很难识别，则跳过这个object
        if cls not in classes or int(difficult) == 1:
            continue
        # 获取类别索引
        cls_id = classes.index(cls)
        # 获取框信息
        xmlbox = obj.find('bndbox')
        # 将框信息转换为(xmin, ymin, xmax, ymax)格式
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        # 向传入的txt中写入 ( ,xmin, ymin, xmax, ymax, cls_id)
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        # 统计不同类别的数量
        nums[classes.index(cls)] = nums[classes.index(cls)] + 1


if __name__ == "__main__":
    # 固定随机种子
    random.seed(0)

    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")

    # 切分数据集为train,val,test，并将对应的图片名写入train.txt, val.txt, test.txt
    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        # 获取所有的xml文件
        xmlfilepath = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
        temp_xml = os.listdir(xmlfilepath)
        total_xml = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)
        # 定义txt的保存路径
        saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')
        # 总共有多少xml文件
        num = len(total_xml)
        # list的数据类型为range(0,num)
        list = range(num)
        # 训练集+验证集数量
        tv = int(num * trainval_percent)
        # 训练集数量
        tr = int(tv * train_percent)
        # 从[0, num)中随机挑选tv个不重复的数字，也就是训练集和验证集的索引
        trainval = random.sample(list, tv)
        # 同上，获取训练集的索引
        train = random.sample(trainval, tr)
        # 打印训练集验证集信息
        print("train and val size", tv)
        print("train size", tr)
        # 打开各自的文件，写入相应的信息
        ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
        ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
        ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
        fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')
        # 遍历[0, num),如果在对应的list中，则将图片名字(不含后缀)写入txt文件
        for i in list:
            name = total_xml[i][:-4] + '\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)
        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()
        print("Generate txt in ImageSets done.")

    # 通过上面生成的train.txt, val.txt, test.txt生成最终的2007_train.txt和2007_val.txt用于训练
    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate 2007_train.txt and 2007_val.txt for train.")
        type_index = 0
        # 遍历[('2007','train'), ('2007', 'val')]
        for year, image_set in VOCdevkit_sets:
            # 读取train.txt并转换为list
            image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set)),
                             encoding='utf-8').read().strip().split()
            list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                # 写入图片路径
                list_file.write('%s/VOC%s/JPEGImages/%s.jpg' % (os.path.abspath(VOCdevkit_path), year, image_id))
                # 每行信息为(img_path ,xmin, ymin, xmax, ymax, cls_id)
                convert_annotation(year, image_id, list_file)
                # 换行
                list_file.write('\n')
            # [train数量, val数量]
            photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()
        print("Generate 2007_train.txt and 2007_val.txt for train done.")

        '''
        用于可视化各类别目标数量
        例：
        |   aeroplane |  1336 | 
        |     bicycle |  1259 | 
        |        bird |  1900 | 
        |        boat |  1294 | 
        |      bottle |  2013 | 
        |         bus |   911 | 
        '''
        def printTable(List1, List2):
            # List1[0]是class列表
            for i in range(len(List1[0])):
                # 这里的print加上end参数是为了不换行，因为默认的print是end='\n'
                print("|", end=' ')
                for j in range(len(List1)):
                    # j取值范围是[0, 1], i取值范围是[0, num_classes)
                    # List1[0]是class列表, List1[1]是每个class对应的数量，这里一行打印类别名和类别对应的数量
                    # List2 = [类别名的最大字符长度, 数量的最大字符长度]
                    # str.rjust(width, str),width是指定的字符串长度，采用右对齐，str为填充内容，默认是空格
                    print(List1[j][i].rjust(int(List2[j])), end=' ')
                    print("|", end=' ')
                # print里面不传入内容，则默认是换行
                print()


        str_nums = [str(int(x)) for x in nums]
        # classes是一个list， str_nums也是一个list
        tableData = [
            classes, str_nums
        ]
        # [0,0]
        colWidths = [0] * len(tableData)
        len1 = 0
        # i取值范围[0,1]
        for i in range(len(tableData)):
            # j取值范围[0, num_classes)
            for j in range(len(tableData[i])):
                if len(tableData[i][j]) > colWidths[i]:
                    # colWidths=[类别名的最大字符长度, 数量的最大字符长度]
                    colWidths[i] = len(tableData[i][j])
        printTable(tableData, colWidths)

        if photo_nums[0] <= 500:
            print("训练集数量小于500，属于较小的数据量，请注意设置较大的训练世代（Epoch）以满足足够的梯度下降次数（Step）。")

        if np.sum(nums) == 0:
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
