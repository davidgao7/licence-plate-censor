import argparse

# import cv2
# from werkzeug.utils import secure_filename

import detect
from experimental import attempt_load
from car_detect.utils import torch_utils

# 检测模型
class Detector(object):

    def __init__(self):
        # 限制输入格式
        self.ALLOW_EXTENSIONS = ['png', 'jpg', 'jpeg']

        # pre-trained model weights
        self.MobileNet_WEIGHT_LAST_PATH = 'car_detect/weights/last.pt'
        self.MobileNet_pt_two_five_Final_PATH = 'car_detect/retinaface/weights/mobilenet0.25_Final.pth'

        # Input output path
        self.CAR_IMAGES_PATH = 'car_detect/inference/images/'
        self.OUTPUT_PATH = 'car_detect/inference/output/'

        # load model
        self.model = self.__init_model__()

    def __init_model__(self):
        # load pre-trained model
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=self.MobileNet_WEIGHT_LAST_PATH,
                            help='model.pt path(s)')
        parser.add_argument('--source', type=str, default=self.CAR_IMAGES_PATH,
                            help='source')  # file/folder, 0 for webcam
        parser.add_argument('--output', type=str, default=self.OUTPUT_PATH, help='output folder')  # output folder
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
        parser.add_argument('--device', default='',
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # mac user has no NVIDIA gpu,
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        self.opt = parser.parse_args()
        self.out, self.source, self.weights, self.view_img, self.save_txt, self.imgsz = \
            self.opt.output, self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
        webcam = self.source == '0' or self.source.startswith('rtsp') or self.source.startswith(
            'http') or self.source.endswith('.txt')
        self.device = torch_utils.select_device(self.opt.device)
        half = self.device.type != 'cpu'  # half precision only supported on CUDA

        model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        print("load pre-trained model success!")
        return model

    def detect(self, image, img_size=640, webcam=0):
        pic_np = image
        pic_censored_np = detect.detect(args=[self.opt], image=image, model=self.model, imgsz=img_size,
                                                    webcam=webcam)  # 车牌识别完成后并打码的图片
        print('returning origin and censored pic...')
        return pic_np, pic_censored_np

