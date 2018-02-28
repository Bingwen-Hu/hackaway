# scrapy project


### start a project
```
scrapy startproject mory-spiders
```

### create spiders
```
scrapy genspider [spider-name] [domain] 
scrapy genspider [spider-name] [domain] -t TEMPLATE
```


### Items
*Items* is the fields to fill in, or the content you are interested in on the website. For example:
```
from scrapy.item import Item, Field

class ProperitesItem(Item):
    # primary fields
    title       = Field()
    price       = Field()
    description = Field()
    address     = Field()
    image_urls  = Field()

    # calculated fields
    images      = Field()
    location    = Field()

    # basic infomation fields
    url         = Field()
    project     = Field()
    spider      = Field()
    server      = Field()
    date        = Field()
```

### start spider
```
scrapy crawl [spider-name]
```
or using the most suitable spider to parse url
```
scrapy parse --spider=[spider-name] url
```

### save to files
save to several format files, including json, j1, csv, xml
```
scrapy crawl [spider-name] -o items.json
```