# mory extension
from scrapy.exceptions import NotConfigured
from scrapy import signals
from twisted.internet import task
import time
class Latencies(object):
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)

    def __init__(self, crawler):
        self.crawler = crawler
        self.interval = crawler.settings.getfloat('LATENCIES_INTERVAL')
        if not self.interval:
            raise NotConfigured

        cs = crawler.signals
        cs.connect(self._spider_opened, signal=signals.spider_opened)
        cs.connect(self._spider_closed, signal=signals.spider_closed)
        cs.connect(self._request_scheduled, signal=signals.request_scheduled)
        cs.connect(self._response_received, signal=signals.response_received)
        cs.connect(self._item_scraped, signal=signals.item_scraped)
        self.latency, self.proc_latency, self.items = 0, 0, 0

    def _spider_opened(self, spider):
        self.task = task.LoopingCall(self._log, spider)
        self.task.start(self.interval)

    def _spider_closed(self, spider, reason):
        if self.task.running:
            self.task.stop()

    def _request_scheduled(self, request, spider):
        request.meta['schedule_time'] = time.time()
    
    def _response_received(self, response, request, spider):
        request.meta['received_time'] = time.time()

    def _item_scraped(self, item, response, spider):
        self.latency += time.time() - response.meta['schedule_time']
        self.proc_latency += time.time() - response.meta['received_time']
        self.items += 1
    
    def _log(self, spider):
        irate = float(self.items) / self.interval
        latency = self.latency / self.items if self.items else 0
        proc_latency = self.proc_latency / self.items if self.items else 0
        spider.logger.info(f"Scraped {self.items} items at {irate} items/s, "
            f"avg latency: {latency} s and avg time in pipelines: {proc_latency} s")
        self.latency, self.proc_latency, self.items = 0, 0, 0