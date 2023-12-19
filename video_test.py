import yolo
import cv2
import os
from PIL import Image
import numpy as np
import torch

yolo = yolo.YOLO()
for file in os.listdir('/media/ly/DATA/Ubuntu/PycharmProjects/car-master/car_detect/inference/images'):
    if file.split('.')[-1] in ['jpg', 'png', 'webp', 'bmp']:
        # box, im0 = find_license(cv2.imread('input/'+file))
        path = '/media/ly/DATA/Ubuntu/PycharmProjects/car-master/car_detect/inference/images/' + file
        im0 = Image.open(path)
        im0 = yolo.detect_image(im0)
        # cv2.rectangle(im0, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imwrite('/media/ly/DATA/Ubuntu/PycharmProjects/car-master/car_detect/inference/output/' + file, np.asarray(im0))
    elif file.split('.')[-1] in ['mp4']:
        cap = cv2.VideoCapture('/media/ly/DATA/Ubuntu/PycharmProjects/car-master/car_detect/inference/images/' + file)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_weight = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap_write = cv2.VideoWriter(
            '/media/ly/DATA/Ubuntu/PycharmProjects/car-master/car_detect/inference/output/' + file,
            cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (int(frame_weight), int(frame_height)))
        is_open = cap.isOpened()
        rect, frame = cap.read()
        while rect:
            rect, frame = cap.read()
            try:
                frame = Image.fromarray(frame)
                im0 = yolo.detect_image(frame)
            except Exception as e:
                print(e)
                break
            cap_write.write(np.asarray(im0))
        cap_write.release()
        cap.release()