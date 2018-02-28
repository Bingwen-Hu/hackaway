# -*- coding: utf-8 -*-
import scrapy
from sina.items import SinaItem

class KeynewsSpider(scrapy.Spider):
    name = 'keynews'
    allowed_domains = ['crawl']
    start_urls = ['http://news.sina.com.cn/']

    def parse(self, response):
        item = SinaItem()
        item['centerNews'] = response.xpath(
            '//*/h1[@data-client="headline"]/a/text()').extract()
        item['rightNews'] =  response.xpath(
            '//*/div[@class="tl"]/a/text()').extract()
        return item
