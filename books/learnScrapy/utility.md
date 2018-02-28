### utility function provide by Scrapy
```
from scrapy.loader.processors import MapCompose, Join
Join()(['hi', 'Join'])
MapCompose(float, int)(['3.14', '2.7'])
MapCompose(lambda i: i.replace(',', ''), float)(['1,234.45'])
```

### unittest in scrapy
create contract in spider keynews:
```
def parse(self, response):
        """ This function parses sina page
        @url http://news.sina.com.cn/
        @returns items 1
        @scrapes centerNews rightNews hostname author
        """
```
then run
```
scrapy check keynews
```