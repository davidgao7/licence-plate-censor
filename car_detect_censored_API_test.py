import os
import time

import cv2
import numpy as np

from car_detect_censored_API import Detector

if __name__ == '__main__':
    detector = Detector()
    # 调用模型
    # image source: https://st.hotrod.com/uploads/sites/21/2015/07/personalized-license-plates-from-mustang-week-2015-31.jpg
    # img = cv2.imread('car_detect/inference/images/personalized-license-plates-from-mustang-week-2015-31-138212729.jpg')
    # t1 = time.time()
    # pic_box_np, pic_censored_np = detector.detect(img)
    # t2 = time.time()
    # print(f'inference time: {t2-t1}')
    # cv2.imshow('box pic',pic_box_np)
    # cv2.imshow('censored pic',pic_censored_np)
    # cv2.waitKey()

    # avg inference time
    times = []
    image_shape = []
    i = 0
    for f in os.listdir('car_detect/inference/images/'):
        if '.jpg' in f:
            img = cv2.imread(
                'car_detect/inference/images/'+f)
            t1 = time.time()
            pic_box_np, pic_censored_np = detector.detect(img)
            t2 = time.time()
            times.append(t2-t1)
            image_shape.append(np.array([img.shape[0], img.shape[1]]))
            i+=1

    avg_img_shape = np.array(image_shape).mean(axis=0)
    print(f'total picture looped: {i} avg inference time: {sum(times)/len(times)}s/inference, avg image shape: {avg_img_shape}')

#[     62.018      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236
      # 34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236      34.236
      # 34.236      34.236      25.127]
# plot one box in [527 306 569 322]
# plot censor in [527 569 306 322]
# plot one box in [215 324 258 338]
# plot censor in [215 258 324 338]
# plot one box in [872 324 906 338]
# plot censor in [872 906 324 338]