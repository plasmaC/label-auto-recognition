from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait

from pyquery import PyQuery as pq
if True:
    # 获取资源
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    browser = webdriver.Chrome(chrome_options=chrome_options)
    browser = webdriver.Chrome()
    wait=WebDriverWait(browser,timeout=10)

    url = "https://s.taobao.com/search?q=手办"
    browser.get(url)



    def index_page(page):
        print("爬取第",page,"页")
        try:
            if page>1:
                # 填入页码并且点击跳转
                inp = wait.until(expected_conditions.presence_of_element_located(
                    (By.CSS_SELECTOR, '#mainsrp-pager div.form > input')
                ))
                submit = wait.until(expected_conditions.element_to_be_clickable(
                    (By.CSS_SELECTOR, '#mainsrp-pager div.form > span.btn.J_Submit')
                ))
                inp.clear()
                inp.send_keys(page)
                submit.click()

            # 等待翻页完成，通过检查以下元素判断
            wait.until(expected_conditions.text_to_be_present_in_element(
                (By.CSS_SELECTOR,'#mainsrp-pager li.item.active > span'),
                str(page)
            ))
            # 加载对应页面的商品
            wait.until(expected_conditions.presence_of_element_located(
                (By.CSS_SELECTOR,'.m-itemlist .items .item')
            ))

            get_goods()

        except TimeoutException:
            print("重新爬取")
            browser.get(url)
            index_page(page)


    def get_goods():

        # pyquery
        html =browser.page_source
        doc=pq(html)
        items = doc('#mainsrp-itemlist .items .item').items()
        for item in items:
            print({
                'img':item.find('.pic .img').attr('data-src'),
                'price': item.find('.price').text(),
                'deal': item.find('.deal-cnt').text(),
                'title':item.find('.title').text(),
                'shop':item.find('.shop').text(),
                'location': item.find('.location').text(),
            })


    for i in range(1,10):
        index_page(page=i)