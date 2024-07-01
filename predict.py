"""
predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
整合到了一个py文件中，通过指定mode进行模式的修改。
"""
import time
import os

import cv2
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

from yolo_predict import YOLO_PREDICT

'''
推理超参设置
'''


def parse_args():
    parser = argparse.ArgumentParser()
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    # ----------------------------------------------------------------------------------------------------------#
    parser.add_argument('--mode', type=str, default='export_onnx',
                        help='predict, video, fps, heatmap, export_onnx')
    parser.add_argument('--model_weight', type=str,
                        default=r'train_result\20230306affine.pth',
                        help='模型权重路径')
    parser.add_argument('--phi', type=str, default='s', help='模型使用的版本')

    parser.add_argument('--save_path', type=str, default='predict_result',
                        help='预测的保存文件夹路径，所有检测结果都会保存在这个路径下')
    parser.add_argument('--classes_path', type=str, default='model_data/cars_classes.txt',
                        help='类别对应的txt文件，一般不修改')
    parser.add_argument('--anchors_path', type=str, default='model_data/yolo_anchors.txt',
                        help='先验框对应的txt文件，一般不修改')
    parser.add_argument('--input_shape', type=list, default=[1152, 2048], help='输入网络的分辨率大小[h,w]')  # [1152, 2048]
    parser.add_argument('--letterbox', type=bool, default=False, help='resize图片时是否使用letterbox')
    parser.add_argument('--confidence', type=float, default=0.1,
                        help='检测结果中只有得分大于置信度的预测框会被保留下来')
    parser.add_argument('--nms_iou', type=float, default=0.3, help='非极大抑制所用到的nms_iou大小')
    parser.add_argument('--crop', type=bool, default=True, help='将检测框裁剪出来并保存,仅在mode="predict"时有效')
    parser.add_argument('--count', type=bool, default=True, help='打印每个类别检测出来的数量,仅在mode="predict"时有效')

    parser.add_argument('--image_path', type=str,
                        default=r'D:\gitlab\cars_detection\datasets\cars_pretrain\images\val',
                        help='检测图片的文件夹路径')
    parser.add_argument('--video_path', type=str, default=r'D:\gitlab\cars_detection\yolov5\video\DJI_0056_test.mp4',
                        help='视频路径，如果为0的话就是调用摄像头,仅在mode="video"时有效')
    parser.add_argument('--video_fps', type=int, default=30, help='用于保存的视频的fps,仅在mode="video"时有效')
    # 视频保存的路径?
    parser.add_argument('--fps_test_interval', type=int, default=10,
                        help='检测网络的fps时连续检测几次取平均,理论上test_interval越大，fps越准确,仅在mode="fps"有效')
    parser.add_argument('--fps_image_path', type=str,
                        default=r'D:\gitlab\zoomtoolkit\datasets\VOC07+12+test\VOCdevkit\VOC2007\JPEGImages\000001.jpg',
                        help='检测网络的fps时所用的图片,仅在mode="fps"有效')
    parser.add_argument('--simplify_onnx', type=bool, default=True, help='简化onnx模型，也就是移除常量算子')
    return parser.parse_args()


def main(args):
    yolo = YOLO_PREDICT(model_path=args.model_weight, classes_path=args.classes_path, anchors_path=args.anchors_path,
                        input_shape=args.input_shape, phi=args.phi, confidence=args.confidence, nms_iou=args.nms_iou,
                        letterbox=args.letterbox)

    # 检测图片
    if args.mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        image_save_path = os.path.join(args.save_path, 'image_output')
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        imgs = os.listdir(args.image_path)
        print("Start Predict...")
        pbar = enumerate(imgs)
        pbar = tqdm(pbar, total=len(imgs))
        for img in pbar:
            img = img[1]
            if img.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(args.image_path, img)
                image = Image.open(image_path).convert('RGB')
                file_name = img.split('.')[0]
                result = yolo.detect_image(image, crop=args.crop, count=args.count, save_path=args.save_path,
                                           file_name=file_name)
                result.save(os.path.join(image_save_path, img.replace(".jpg", ".png")), quality=95, subsampling=0)
            pbar.update(1)
        pbar.close()
        print("Finish Predict.")

    # 检测视频
    elif args.mode == "video":
        start = time.time()
        print('------开始计算------')
        capture = cv2.VideoCapture(args.video_path)
        video_save_folder = os.path.join(args.save_path, 'video_output')
        if not os.path.exists(video_save_folder):
            os.makedirs(video_save_folder)
        video_save_path = os.path.join(video_save_folder, os.path.basename(args.video_path))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, args.video_fps, size)
        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(yolo.detect_image(frame, save_path=args.save_path))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # 这里是获取检测的fps，并显示在图片上
            fps = (fps + (1. / (time.time() - t1))) / 2
            # print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # 展示图片(注释掉则不需要展示)
            # cv2.imshow("video", frame)
            # c = cv2.waitKey(1) & 0xff  # 展示的图片等待1ms
            if video_save_path != "":
                out.write(frame)
            # Esc的ASCII码为27，表示如果按下esc，则停止检测
            # if c == 27:
            #     capture.release()
            #     break

        print("Video Detection Done!")
        capture.release()
        print("Save processed video to the path :" + video_save_path)
        out.release()
        cv2.destroyAllWindows()
        end = time.time()
        print('------计算完成------')
        print(f'计算总时长为: {end - start}s')

    # 测试网络的fps
    elif args.mode == "fps":
        img = Image.open(args.fps_image_path)
        tact_time = yolo.get_FPS(img, args.fps_test_interval)
        # print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')
        print('batch_size: 1; \naverage predict time: {:.5f}s; \nFPS: {:.3f}'.format(tact_time, (1 / tact_time)))

    # 检测热力图
    elif args.mode == "heatmap":
        heatmap_save_path = os.path.join(args.save_path, 'heatmap_output')
        if not os.path.exists(heatmap_save_path):
            os.makedirs(heatmap_save_path)
        imgs = os.listdir(args.image_path)
        for img in imgs:
            if img.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(args.image_path, img)
                image = Image.open(image_path).convert('RGB')
                yolo.detect_heatmap(image, heatmap_save_path)

    # 将模型转换成onnx格式
    elif args.mode == "export_onnx":
        onnx_save_folder = os.path.join(args.save_path, 'onnx_model')
        if not os.path.exists(onnx_save_folder):
            os.makedirs(onnx_save_folder)
        onnx_save_path = os.path.join(onnx_save_folder, 'model.onnx')
        yolo.convert_to_onnx(args.simplify_onnx, onnx_save_path)

    else:
        raise AssertionError(
            "Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx'.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
