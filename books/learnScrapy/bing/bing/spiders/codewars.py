# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy.http import FormRequest

class CodewarsSpider(CrawlSpider):
    name = 'codewars'
    allowed_domains = ['codewars']
    
    # replace start_urls
    def start_requests(self):
        formrequest = FormRequest(
            "https://www.codewars.com/users/sign_in",
            formdata={"user[email]": "user@mail",
                      "user[password]": "password"}
        )
        # wrap in a list
        return [formrequest]

    def parse(self, response):
        return {"url": response.url}