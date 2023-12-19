# -*- coding:UTF-8 -*-
# imports
import json
import sys
from flask import Flask, request, app, jsonify
from werkzeug.utils import secure_filename  # 去中文
import os
import cv2
import datetime
from car_detect_censored_API import Detector

# initial port
app = Flask(__name__)

# 定义存储图片地址
app.config['SAVED_PIC'] = 'car_image'
app.config['SAVED_PIC_CENSORED'] = 'car_image_censored'

ALLOW_EXTENSIONS = ['png', 'jpg', 'jpeg']


def is_allowed_file(filename):
    return isinstance(filename, str) and '.' in filename and filename.rsplit('.', 1)[
        -1].lower() in ALLOW_EXTENSIONS


@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    detector = Detector()  # 引用模型

    if request.method == 'POST':

        # 返回原图片和加密图片的json格式
        file = request.files['file']  # 一个文件, key: 'file'
        print("request success")

        # 判断 文件格式
        if file and is_allowed_file(file.filename):
            field_name = secure_filename(file.filename)  # 确保没有中文字符 （cv2 读不了中文字符）
            pic_path = os.path.join(app.config['SAVED_PIC'], field_name)  # 图片文件地址

            # save original pic
            file.save(pic_path)
            print('pic save to %s' % pic_path)

            # 调用模型
            img = cv2.imread(pic_path)
            pic_np, pic_censored_np = detector.detect(img)  # get the original pic and censored pic
            pic_dic = {}
            pic_dic['original'] = pic_np.tolist()
            pic_dic['censored'] = pic_censored_np.tolist()
            # save censored version pic
            censored_pic_path = os.path.join(app.config['SAVED_PIC_CENSORED'],
                                             field_name.split('.')[0] + '_censored.' + field_name.split('.')[-1])
            cv2.imwrite(censored_pic_path, pic_censored_np)
            print('censored pic saved to %s' % censored_pic_path)

            return jsonify(
                {'image_url': 'http://172.16.162.138:1111/upload/origin_' + field_name,
                 'image_censored_url': 'http://172.16.162.138:1111/upload/censored_' + field_name,
                 'picture': pic_dic,  # dic['index']
                 }
            )

    else:
        print('get')
        return jsonify({'status': 0})


def run():
    app.run(host='0.0.0.0', debug=True, port=8000, threaded=True)


if __name__ == '__main__':
    run()
