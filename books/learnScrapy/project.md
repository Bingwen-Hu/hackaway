# scrapy project


### start a project
```
scrapy startproject mory-spiders
```

### create spiders
```
scrapy genspider [spider-name] [domain]
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
scrapy crawl [project-name]
```