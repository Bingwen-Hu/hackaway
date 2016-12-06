"""
This script handle the "nonce" when log in github.com 
special thanks to tha author of <learning scrapy>
"""


from scrapy.spiders import CrawlSpider
from scrapy.http import Request, FormRequest

class GithubLoginSpider(CrawlSpider):
    name = 'githublogin'

    # Start on the welcome page
    def start_requests(self):
        return [
            Request(
                "https://github.com/login",
                callback=self.parse_welcome)
        ]

    # Post welcome page's first form with the given user/pass
    def parse_welcome(self, response):
        return FormRequest.from_response(
            response,
            formdata={"login": "yourUsername", "password": "yourPassword"}
        )

    def parse(self, response):
        # of course, you can do more!
        h2 = response.xpath('//*[@id="js-pjax-container"]//h2').extract_first()
        print(h2)
        
