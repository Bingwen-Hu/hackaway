### utility function provide by Scrapy
```
from scrapy.loader.processors import MapCompose, Join
Join()(['hi', 'Join'])
MapCompose(float, int)(['3.14', '2.7'])
MapCompose(lambda i: i.replace(',', ''), float)(['1,234.45'])
```