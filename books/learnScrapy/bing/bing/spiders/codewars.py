# -*- coding: utf-8 -*-
""" testing login using scrapy
1. start_reqeust do the init fetch, get the form data back
2. parse_login then fill the user and pass field in the form,
   by doing this, url change to dashboard, nor sign_in
3. parse the dashboard page


Note that, run from command line and pass the user and password as
$ scrapy crawl codewars -a username=[username] -a password=[password]
"""

import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy.http import FormRequest, Request

class CodewarsSpider(CrawlSpider):
    name = 'codewars'
    allowed_domains = ['www.codewars.com']
    
    # replace start_urls
    def start_requests(self):
        request = Request(
            "https://www.codewars.com/users/sign_in",
            callback=self.parse_login
        )
        # wrap in a list
        return [request]

    def parse_login(self, response):
        """using formrequest to copy all other fields
        from the original response
        """
        formrequest = FormRequest.from_response(
            response, 
            formdata={"user[email]"    : getattr(self, 'username'),
                      "user[password]" : getattr(self, 'password')}
        )
        return formrequest

    def parse(self, response):
        return {"url": response.url}