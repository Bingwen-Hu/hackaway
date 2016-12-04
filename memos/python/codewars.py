# I got an idea. Run a script and get a kata in codewars
# saving opening up the chrome and so and so on ... :D


from scrapy.spiders import CrawlSpider
from scrapy.http import Request, FormRequest

class GithubLoginSpider(CrawlSpider):
    name = 'codewars'

    # Start on the welcome page
    def start_requests(self):
        return [
            Request(
                "https://www.codewars.com/users/sign_in",
                callback=self.parse_welcome)
        ]

    # Post welcome page's first form with the given user/pass
    def parse_welcome(self, response):
        return FormRequest.from_response(
            response,
            formdata={"user[email]": "username",
                      "user[password]": "password"}
        )

    def parse(self, response):
        # of course, you can do more!
        # it seems that the LANGuage is also a param!
        yield {
            'kata-url': response.xpath('//*[@id="personal_trainer_next_challenge"]').extract_first()
        }
