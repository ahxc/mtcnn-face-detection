import warnings
warnings.filterwarnings('ignore')

import os
import sys

import cv2
import matplotlib.pyplot as plt

import tensorflow as tf

# 导入mtcnn
import detect_face

#----------------------------------------------------------------------------
# 相关路劲设置

# 根目录
ROOT_DIR = os.getcwd()
# 数据目录
DATA_PATH = os.path.join(ROOT_DIR, "data")
# 测试图像目录
IMGS_PATH = os.path.join(DATA_PATH, "images")
# 侦测结果目录
RESULTS_PATH = os.path.join(DATA_PATH, "results")
# 裁剪结果目录
CROP_PATH = os.path.join(RESULTS_PATH, "img_crop")

# 获得输入图像文件名
try:
    TEST_IMAGE = sys.argv[1]
except:
    TEST_IMAGE = "gameofthrones.png"

#----------------------------------------------------------------------------
# 相关函数定义

# 读图函数
def imread(filename):
    
    return cv2.imread(os.path.join(IMGS_PATH, filename))

# 裁剪图像保存
def crop_outfiles(img, img_name, i):
    
    path = os.path.join(CROP_PATH, "{}".format(img_name.split('.')[0]))
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    outfile_path = os.path.join(path, "{}-{}".format(str(i), img_name))
    cv2.imwrite(outfile_path, img)

# 侦测图像保存
def detection_outfiles(img, img_name):

    outfile_path = os.path.join(RESULTS_PATH, "{}".format(img_name))
    cv2.imwrite(outfile_path, img)

#----------------------------------------------------------------------------
# 设定网络参数

# 最小人脸尺寸
minsize = 30
# P-net，R-net，O-net阈值，即人脸置信度最低阈值
threshold = [ 0.6, 0.7, 0.7 ]
# 图像金字塔比例因子，即图像逐步缩放的比例因子
# 所有scale作为三个网络的输入，为了使网络能检测到不同scale的图像中的人脸
factor = 0.709

gpu_memory_fraction = 1.0

#----------------------------------------------------------------------------
# 构建MTCNN图

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = gpu_memory_fraction)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options, log_device_placement = False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

#----------------------------------------------------------------------------
# 操作开始

# 读图并调整通道顺序
bgr_image = imread(TEST_IMAGE)
rgb_image = bgr_image[:,:,::-1]

# 人脸侦测
bounding_boxes, _ = detect_face.detect_face(rgb_image, minsize, pnet, rnet, onet, threshold, factor)

# 复制输入图像
draw = bgr_image.copy()

# box个数，即侦测到的人脸个数
faces_detected_number = len(bounding_boxes)
print('Total faces detected :{}'.format(faces_detected_number))

# 设置一个list存储所有侦测到的人脸 
crop_list = []
# 裁剪人脸计数器
i = 1

# 每一个bounding_box包括了(x1, y1, x2, y2, 置信度)
# 迭代每一个bounding_
for face_position in bounding_boxes:
    # 边框左边只接受整型
    face_position = face_position.astype(int)

    # 左上角坐标(x1, y1)，右下角坐标(x2, y2)
    # 由于图像边界的人脸可能会侦测出负值
    # 进行数值修正
    x1 = max(face_position[0], 0)
    y1 = max(face_position[1], 0)
    x2 = max(face_position[2], 0)
    y2 = max(face_position[3], 0)

    # 在复制品上画边界框    
    cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 提取边界框内的图像(可用于其他模块)
    crop = bgr_image[y1:y2, x1:x2,]
    # 每个框设置统一尺寸
    crop = cv2.resize(crop, (96, 96), interpolation = cv2.INTER_CUBIC)
    # 加入list
    crop_list.append(crop)

    # 存储裁剪图像
    crop_outfiles(crop, TEST_IMAGE, i)
    i += 1

    # 临时展示
    # plt.imshow(crop)
    # plt.show()

# 输出画外边界框的图像
detection_outfiles(draw, TEST_IMAGE)

#----------------------------------------------------------------------------
