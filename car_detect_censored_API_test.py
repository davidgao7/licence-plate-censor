import cv2

from car_detect_censored_API import Detector

if __name__ == '__main__':
    detector = Detector()
    # 调用模型
    img = cv2.imread('car_detect/inference/images/personalized-license-plates-from-mustang-week-2015-31-138212729.jpg')
    pic_box_np, pic_censored_np = detector.detect(img)
    cv2.imshow('box pic',pic_box_np)
    cv2.imshow('censored pic',pic_censored_np)
    cv2.waitKey()

# plot one box in [527 306 569 322]
# plot censor in [527 569 306 322]
# plot one box in [215 324 258 338]
# plot censor in [215 258 324 338]
# plot one box in [872 324 906 338]
# plot censor in [872 906 324 338]