# -*- coding: utf-8 -*-

import scrapy
from scrapy.http import Request
from scrapy.loader import ItemLoader
from scrapy.loader.processors import MapCompose, Join
from datetime import datetime
import urllib.parse as urlparse

from bing.items import BingItem


class CrossSpider(scrapy.Spider):
    name = 'cross'
    allowed_domains = ['doc.scrapy.org']
    start_urls = ['https://doc.scrapy.org/en/latest/intro/overview.html']

    def parse_item(self, response):
        loader = ItemLoader(BingItem(), response=response)
        loader.add_xpath('Title', "//*/h1/text()")
        loader.add_value('Date', datetime.now())
        return loader.load_item()

    def parse(self, response):
        # get next url
        next_selector = response.xpath('//*/li[@class="toctree-l1"]/a/@href')

        # collect all urls
        for url in next_selector.extract():
            yield Request(urlparse.urljoin(response.url, url),
                    callback=self.parse_item)