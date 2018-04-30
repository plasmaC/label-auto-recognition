import json
import os
import re
import socket
import time
import urllib
import urllib.error
import urllib.parse
import urllib.request

# 设置超时
timeout = 2
socket.setdefaulttimeout(timeout)


# 获取referrer，用于生成referrer
def get_referrer(url):
    par = urllib.parse.urlparse(url)
    if par.scheme:
        return par.scheme + '://' + par.netloc
    else:
        return par.netloc


# 获取后缀名
def get_suffix(name):
    m = re.search(r'\.[^.]*$', name)
    if m.group(0) and len(m.group(0)) <= 5:
        return m.group(0)
    else:
        return '.jpeg'


class Crawler:
    # 睡眠时长
    __time_sleep = 0.1
    __amount = 0
    __start_amount = 0

    # 成功保存的图片数
    __counter = 0

    # 设置header防ban
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}

    # t 下载图片时间间隔
    def __init__(self, t):
        self.time_sleep = t

    # 保存图片
    def save_image(self, response_data, word):
        # 建立文件夹
        if not os.path.exists("./" + word):
            os.mkdir("./" + word)

        # 判断名字是否重复，获取图片长度
        self.__counter = len(os.listdir('./' + word)) + 1
        for image_info in response_data['imgs']:

            try:
                time.sleep(self.time_sleep)
                suffix = get_suffix(image_info['objURL'])

                # 指定User-agent和referrer，减少403
                referrer = get_referrer(image_info['objURL'])
                opener = urllib.request.build_opener()
                opener.addheaders = [
                    ('User-agent', 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:55.0) Gecko/20100101 Firefox/55.0'),
                    ('Referrer', referrer)
                ]
                urllib.request.install_opener(opener)

                # 保存图片
                urllib.request.urlretrieve(image_info['objURL'], './' + word + '/' + str(self.__counter) + str(suffix))

            except urllib.error.HTTPError as urllib_err:
                print(urllib_err)
                continue

            except Exception as err:
                time.sleep(1)
                print(err)
                print("出现错误，放弃保存")
                continue

            else:
                print("已保存" + str(self.__counter) + "/" + str(self.__amount) + "张图")
                self.__counter += 1
        return

    # 开始获取
    def get_images(self, word):
        search = urllib.parse.quote(word)
        # pn int 图片数
        pn = self.__start_amount
        while pn < self.__amount:

            url = 'http://image.baidu.com/search/avatarjson?tn=resultjsonavatarnew&ie=utf-8&word=' + search \
                  + "&cg=girl&pn=" + str(pn) + '&rn=60&itg=0&z=0&fr=&width=&height=&lm=-1&ic=0&s=0&st=-1&gsm=1e0000001e'
            try:
                time.sleep(self.time_sleep)
                request = urllib.request.Request(url=url, headers=self.headers)
                page = urllib.request.urlopen(request)
                response = page.read().decode('unicode_escape')
            except UnicodeDecodeError as e:
                print(e)
                print('-----UnicodeDecodeErrorUrl:', url)
            except urllib.error.URLError as e:
                print(e)
                print("-----urlErrorUrl:", url)
            except socket.timeout as e:
                print(e)
                print("-----socket.timeout:", url)
            else:
                # 解析json
                response_data = json.loads(response)
                self.save_image(response_data, word)

                # 读取下一页
                print("加载下一页")
                pn += 60
            finally:
                page.close()
        print("下载任务结束")
        return

    def start(self, word, spider_page_num, start_page):
        """
        爬虫入口
        :param word: 抓取的关键词
        :param spider_page_num: 需要抓取数据页数 总抓取图片数量为 页数x60
        :param start_page:起始页数
        :return:
        """
        self.__start_amount = (start_page - 1) * 60
        self.__amount = spider_page_num * 60 + self.__start_amount
        self.get_images(word)


if __name__ == '__main__':
    crawler = Crawler(0.05)  # 抓取延迟为 0.05

    crawler.start('iPhone', 10, 1)
