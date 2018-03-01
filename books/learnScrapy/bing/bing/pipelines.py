# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

from datetime import datetime

class BingPipeline(object):
    def process_item(self, item, spider):
        return item

class TidyUp(object):
    def process_item(self, item, spider):
        if spider.name == 'cross':
            item['Date'] = [date.isoformat() for date in item['Date']]
        return item