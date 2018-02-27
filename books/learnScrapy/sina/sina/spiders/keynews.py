# -*- coding: utf-8 -*-
import scrapy


class KeynewsSpider(scrapy.Spider):
    name = 'keynews'
    allowed_domains = ['crawl']
    start_urls = ['http://news.sina.com.cn/']

    def parse(self, response):
        self.log("keynews: %s" % response.xpath(
            '//*/h1[@data-client="headline"][1]/a/text()').extract_first())
        self.log("right page: %s" % response.xpath(
            '//*/div[@class="tl"]/a/text()').extract_first())
