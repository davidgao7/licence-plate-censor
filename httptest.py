import requests
import json
import os

files = {'file': open(r'/Users/tengjungao/car-master/car_detect/inference/images/1.png', 'rb')}

# res = requests.post(url="http://172.16.162.195:1111/upload", files=files)
res = requests.post(url="http://8.142.121.161:8888/upload", files=files)

Text = json.loads(res.text)
print(Text)

# test_one_pic = True
#
# for pic_root, pic_dir, pic_files in os.walk('/Users/tengjungao/car-master/car_detect/inference/images/'):
#
#     for i, pic_file in enumerate(pic_files):
#             files_dic = {
#                 'file':
#                     open(pic_root+pic_file, 'rb')
#             }
#
#             res = requests.post(url="http://172.16.162.138:1111/upload", files=files_dic)  # 车牌
#             Text = json.loads(res.text)
#
#             # print(Text)
#             if test_one_pic:
#                 break
#
#     if test_one_pic:
#         break
