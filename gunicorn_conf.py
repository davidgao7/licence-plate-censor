# gunicorn配置文件

workers = 5    # 定义同时开启的处理请求的进程数量，根据网站流量适当调整
worker_class = "gevent"   # 采用gevent库，支持异步处理请求，提高吞吐量
bind = "172.16.162.138:1111"

# 需要增加如下代码，gunicorn才会输出运行日志
accesslog = './logs/gunicorn.access.log'
accesslog = '-' # 记录到标准输出
errorlog = './logs/gunicorn.error.log'
errorlog = '-' # 记录到标准输出
