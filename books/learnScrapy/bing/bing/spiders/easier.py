# -*- coding: utf-8 -*-
import scrapy
from scrapy.loader import ItemLoader
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from bing.items import BingItem
from datetime import datetime

class EasierSpider(CrawlSpider):
    name = 'easier'
    allowed_domains = ['doc.scrapy.org']
    start_urls = ['https://doc.scrapy.org/en/latest/intro/overview.html']

    # here, a or href is discarded
    rules = (
        Rule(LinkExtractor(restrict_xpaths='//*/li[@class="toctree-l1"]'), callback='parse_item'),
    )

    def parse_item(self, response):
        loader = ItemLoader(BingItem(), response=response)
        loader.add_xpath('Title', "//*/h1/text()")
        loader.add_value('Date', datetime.now())
        return loader.load_item()
