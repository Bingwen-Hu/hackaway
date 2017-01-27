# Alibaba Tichi Big data competition 
# dataset for future download: https://tianchi.shuju.aliyun.com/datalab/
# at this time, dataset is in the competition page.

# ==================== data analysis
# shop_info 2000 entries
# field: shop_id, city_name, location_id, per_pay, score, comment_cnt, shop_level, 
# cate_1_name, cate_2_name, cate_3_name

# user_pay => positive examples
# field: user_id, shop_id, time_stamp

# user_view => negative examples
# field: user_id, shop_id, time_stamp

# target: for every shop we should predict its 14 days consumption
# common sense
# different city has different consumption mode
# people tend to consume the same thing in the same shop if they like.


import pandas as pd
import numpy as np

shopInfo_headers = ['shop_id', 'city_name', 'location_id', 
           'per_pay', 'score', 'comment_cnt', 'shop_level', 
           'cate_1_name', 'cate_2_name', 'cate_3_name']
user_headers = ['user_id', 'shop_id', 'time_stamp']

shop_info = pd.read_csv('shop_info.txt', names=shopInfo_headers)
user_pay = pd.read_csv("user_pay.txt", names=user_headers, nrows=100000) # the data is too big
user_view = pd.read_csv("user_view.txt", names=user_headers)
extra_user_view = pd.read_csv("extra_user_view.txt", names=user_headers)

print("shop_info, user_pay, user_view, extra_user_view: ")
print(shop_info.shape, user_pay.shape, user_view.shape, extra_user_view.shape)

# shop_info, user_pay, user_view, extra_user_view: 
# (2000, 10) (100000, 3) (5556715, 3) (4549929, 3)

cities = shop_info['city_name'] # only 122 cities
locations = shop_info['location_id']

# locations.describe()

# count    2000.000000
# mean      583.083000
# std       335.763357
# min         1.000000
# 25%       287.750000
# 50%       577.500000
# 75%       877.250000
# max      1159.000000
# Name: location_id, dtype: float64

# cities.describe()

# count     2000
# unique     122
# top         上海
# freq       285
# Name: city_name, dtype: object
