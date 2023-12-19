from models.experimental import attempt_load
from utils.grneral import non_max_suppression, scale_coords
from utils.utils import *
from utils.torch_utils import select_device
import cv2
from random import randint
from car_detect.utils.datasets import letterbox


class FaceDetector(object):

    def __init__(self, img_size):
        self.img_size = img_size
        self.threshold = 0.4
        self.initial_model()

    def initial_model(self, yolo_version='5m'):
        if yolo_version == '5l':
            self.weights = '/Users/tengjungao/car-master/car_detect/retinaface/weights/yolov5l.pt'
        elif yolo_version == '5m':
            self.weights = '/Users/tengjungao/car-master/car_detect/retinaface/weights/yolov5m.pt'
        elif yolo_version == '5s':
            self.weights = '/Users/tengjungao/car-master/car_detect/retinaface/weights/yolov5s.pt'
        elif yolo_version == '5x':
            self.weights = '/Users/tengjungao/car-master/car_detect/retinaface/weights/yolov5x.pt'

        device = str(torch.cuda.device_count()) if torch.cuda.is_available() else 'cpu'
        self.device = select_device(device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        model.half()
        self.model = model
        self.names = model.module.names if hasattr(model, 'module') else model.names
        self.colors = [(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in self.names]

    def preprocess(self, img):
        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def plot_bboxes(self, img, bboxes, line_thickness=None):
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        for (x1, y1, x2, y2, cls_id, conf) in bboxes:
            color = self.colors[self.names.index(cls_id)]
            c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

        return img

    def detect(self, img):
        im0, img = self.preprocess(img)

        pred = self.model(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.3)

        pred_boxes = []
        count = 0

        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:],
                                          det[:, :4],
                                          im0.shape).round()

                for *x, conf, cls_id in det:
                    labels = self.names[int(cls_id)]
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])

                    pred_boxes.append(
                        (x1, y1, x2, y2, labels, conf)
                    )
                    count += 1

        img = self.plot_bboxes(img, pred_boxes)
        return img
