import matplotlib.pyplot as plt

from car_detect.models.experimental import *
from car_detect.utils.datasets import *
from car_detect.utils.utils import *
from car_detect.models.LPRNet import *
from car_detect.retinaface.test_fddb import fdd_detect, load_model
# from models.yolo import Model
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from car_detect.retinaface.data import cfg_mnet, cfg_re50
import cv2
from car_detect.retinaface.models.retinaface import RetinaFace

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Data File Locations:
MobileNet_WEIGHT_LAST_PATH = 'car_detect/weights/last.pt'
MobileNet_pt_two_five_Final_PATH = 'car_detect/retinaface/weights/mobilenet0.25_Final.pth'
CAR_IMAGES_PATH = 'car_detect/inference/images/'
OUTPUT_PATH = 'car_detect/inference/output/'


def detect(image, model, imgsz,webcam, args=None, device=None, net=None, cfg=None ):
    save_img = True
    img = letterbox(image, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    dataset = [(img, image)]
    # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.names

    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    colors = [255, 0, 0]
    # model = model.cpu()
    # with torch.no_grad():
    #     input = torch.rand([1, 3, 640, 640]).float()
    #     onnx_path = '/media/ly/DATA/Ubuntu/VScodeProject/last.onnx'
    #     torch.onnx.export(model.cpu(), input, onnx_path, verbose=True, input_names=['input'], output_names=['output'])



    # Run inference
    # t0 = time.time()
    # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for img, im0s in dataset:
        img = torch.from_numpy(img).float()
        # img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if args[0] is not None:
            argument = args[0].augment
            opt = args[0]
        else:
            argument = opt.augment

        # Inference
        # t1 = torch_utils.time_synchronized()
        pred = model(img, augment=argument)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # t2 = torch_utils.time_synchronized()

        # Apply Classifier
        # if classify:
        #     pred,plat_num = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                s, im0 = '%g: ' % i, im0s[i].copy()
            else:
                s, im0 = '', im0s

            img_censored = None
            # save_path = str(Path(out) / Path(p).name)
            # txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    # s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for de in det:
                    # xyxy,conf,cls,lic_plat=de[:4],de[4],de[5],de[6:]
                    *xyxy, conf, cls=de

                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * 5 + '\n') % (cls, xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        # label = '%s %.2f' % (names[int(cls)], conf)
                        # label = '%s %.2f' % (lb, conf)
                        label = ''
                        # add one box in the image
                        im0=plot_one_box(xyxy, im0, label=label, color=colors, line_thickness=3, censord=True)

                        # add one censored area in the copy image
                        efficient = True

                        # img_censored=plot_censor(xyxy, im0, efficient)  # 车牌打码
                        # img_censored=face_censor(img,efficient)  # TODO:人脸打码

                        # im0=fdd_detect(im0, args, device, net, cfg)
                        # im0=detect_mtcnn(im0)
                        # im0=cv2_detect(im0)
                        # print(type(im0))
                        #     img_pil = Image.fromarray(im0) # narray转化为图片
                        #     im0 = yolo.detect_image(img_pil) #图片才能检测
            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, t2 - t1))#不打印东西

            # Stream results
            # if view_img:
            #     cv2.imshow(p, im0)
            #     if cv2.waitKey(1) == ord('q'):  # q to quit
            #         raise StopIteration

            # Save results (image with detections)
            im0 = np.array(im0)  # 图片转化为 narray
            # img_censored = np.array(img_censored)
            # cv2.imwrite('/media/ly/DATA/Ubuntu/PycharmProjects/car-master/inference/output/b19c924d4152a90ad3578c66d4400fac.jpeg', im0) #这个地方的im0必须为narray
            return im0
                # else:
                #     if vid_path != save_path:  # new video
                #         vid_path = save_path
                #         if isinstance(vid_writer, cv2.VideoWriter):
                #             vid_writer.release()  # release previous video writer
                #
                #         fourcc = 'mp4v'  # output video codec
                #         fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #         w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #         h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                #     im0 = np.array(im0) # 图片转化为 narray#JMW添加
                #     vid_writer.write(im0)

    # if save_txt or save_img:
    #     # print('Results saved to %s' % os.getcwd() + os.sep + out)
    #     if platform == 'darwin':  # MacOS
    #         os.system('open ' + save_path)

    # print('Done. (%.3fs)' % (time.time() - t0))#不打印一堆东西


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=MobileNet_WEIGHT_LAST_PATH, help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=CAR_IMAGES_PATH, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default=OUTPUT_PATH, help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # mac user has no nivida gpu,
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()

    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    device = torch_utils.select_device(opt.device)

    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model

    print(model)

    #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    # if half:
    #     model.half()  # to FP16

    # Second-stage classifier
    # classify = True
    # if classify:
    #     # modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
    #     # modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
    #     modelc = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(device)
    #     modelc.load_state_dict(torch.load('/media/ly/DATA/Ubuntu/PycharmProjects/car-master/car_detect/weights/Final_LPRNet_model.pth', map_location=torch.device('cpu')))
    #     print("load pretrained model successful!")
    #     modelc.to(device).eval()

    parser = argparse.ArgumentParser(description='Retinaface')

    parser.add_argument('-m', '--trained_model',
                        default=MobileNet_pt_two_five_Final_PATH,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--dataset', default='FDDB', type=str, choices=['FDDB'], help='dataset')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')

    # for mac with non-CUDA gpu
    if device.type == 'cpu':
        args.cpu = True

    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    with torch.no_grad():
        # if opt.update:  # update all models (to fix SourceChangeWarning)
        #     for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
        #         detect()
        #         create_pretrained(opt.weights, opt.weights)
        # else:

        for file in os.listdir(CAR_IMAGES_PATH):  # for every file in the dir file list
            if file.split('.')[-1] in ['jpg', 'png', 'webp', 'bmp']:
                # box, im0 = find_license(cv2.imread('input/'+file))
                path = CAR_IMAGES_PATH + file
                im0 = cv2.imread(path)
                cv2.imwrite(OUTPUT_PATH + file, im0)
                im0 = detect(im0, imgsz, args, device, net, cfg, model)  # get found licence plate number and its censored version
                # cv2.rectangle(im0, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.imwrite(OUTPUT_PATH + file.split('.')[0] + '_censored.' + file.split('.')[-1], img_censored)
            elif file.split('.')[-1] in ['mp4']:
                cap = cv2.VideoCapture(CAR_IMAGES_PATH + file)
                frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                frame_weight = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap_write = cv2.VideoWriter(OUTPUT_PATH+file, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (int(frame_weight), int(frame_height)))
                is_open = cap.isOpened()
                rect, frame = cap.read()
                while rect:
                    rect, frame = cap.read()
                    try:
                        im0 = detect(frame, args, device, net, cfg, model)
                    except Exception as e:
                        print(e)
                        break
                    cap_write.write(im0)  # img_censored if you want the licence censored version
                cap_write.release()
                cap.release()

            # 试运行一个图
            print("1 pic done")
            break
