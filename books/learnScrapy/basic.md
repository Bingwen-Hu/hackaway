### start scrapy shell without debug infomation
```
scrapy shell --nolog
```

### basic, in scrapy shell
note that xpath subscript is start from 1 not 0
```
url = 'http://news.sina.com.cn'
fetch(url)
response.xpath("//*/a/text()").extract()
response.xpath('//*/a[@href="http://blog.sina.com.cn/lm/sports/"]').extract()
response.xpath('//*/a/span[@class="ct_tit"][1]/text()).extract()
```