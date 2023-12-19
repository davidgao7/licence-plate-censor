# 基础镜像
# A series of Docker images that allows you to quickly set up your deep learning research environment.
FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    python \
    python3 \
	python-dev \
	python3-dev \
    adb \
    vim \
    python3-pip \
    unzip \
    sudo \
    wget \
    apt-utils \
	libc++-9-dev \
    zip

RUN apt-get update && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1 && \
    update-alternatives --config python

#  定义工作目录
WORKDIR /car-master
COPY requirements.txt ./

RUN pip3 install --upgrade pip && \
	pip3 install -r requirements.txt -i https://pypi.douban.com/simple/ && \
    pip3 install torch==1.9.0 torchvision==0.10.0 -i https://pypi.douban.com/simple/ && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# dockerfile 指令每执行一次都会在docker上新建一层, 过多无意义的层，会造成镜像膨胀过大,
# e.g. 3-layer image--> 1-layer image in this case
COPY . .

CMD ["gunicorn", "car_detect_censored_flask:app", "-c", "./gunicorn_conf.py"]
