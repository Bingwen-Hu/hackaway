### start scrapy shell without debug infomation
```
scrapy shell --nolog
```

### basic, in scrapy shell
```
url = 'http://news.sina.com.cn'
fetch(url)
response.xpath("//*/a/text()").extract()
response.xpath('//*/a[@href="http://blog.sina.com.cn/lm/sports/"]').extract()
response.xpath('//*/a/span[@class="ct_tit"]/text()).extract()
```