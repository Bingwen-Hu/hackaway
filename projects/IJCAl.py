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

# ==================== what should be excluded and included?
# shop_id sbould NOT be in training because it is just identifies shops.
# city_name is an important feature!
# location_id will confuse the model so should NOT be!
# per_pay shows the price should be in.
# numeric: score, comment_cnt, shop_level may be important.
# cate_1, cate_2, cate_3 are categories messages. Very important.should be ENCODED

# so train features are: 
# city_name (decode) 122
# per_pay
# score
# comment_cnt
# shop_level
# cate_1 (decode)  6 
# cate_2 (decode)  17
# cate_3 (decode)  44


# shop_info, user_pay, user_view, extra_user_view: 
# (2000, 10) (100000, 3) (5556715, 3) (4549929, 3)

cities = shop_info['city_name'] # only 122 cities
locations = shop_info['location_id']
cate_1 = shop_info['cate_1_name']
cate_2 = shop_info['cate_2_name']
cate_3 = shop_info['cate_3_name']
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

# cate_1

# 0             美食
# 1          超市便利店
# 68          休闲娱乐
# 246         医疗健康
# 493     美发/美容/美甲
# 1756          购物

# cate_2

# 0              休闲茶饮
# 1                超市
# 4              休闲食品
# 5              烘焙糕点
# 6                快餐
# 10               小吃
# 11               中餐
# 17               火锅
# 19      汤/粥/煲/砂锅/炖菜
# 22              便利店
# 23             其他美食
# 68             网吧网咖
# 100              烧烤
# 246              药店
# 493            美容美发
# 1756           本地购物
# 1846           个人护理

# cate_3

# 0          饮品/甜点
# 1            NaN
# 2             奶茶
# 4           生鲜水果
# 5             面包
# 6           西式快餐
# 10          其它小吃
# 11           东北菜
# 12          中式快餐
# 17       麻辣烫/串串香
# 19             粥
# 20            蛋糕
# 23            西餐
# 28         米粉/米线
# 43       川味/重庆火锅
# 44            川菜
# 60            面点
# 63           冰激凌
# 73          其它快餐
# 76           咖啡厅
# 81            粤菜
# 93        其它烘焙糕点
# 100         中式烧烤
# 116          江浙菜
# 151           零食
# 163     砂锅/煲类/炖菜
# 165         日韩料理
# 168          西北菜
# 183        其它地方菜
# 184       其它休闲食品
# 252           海鲜
# 313           咖啡
# 325         其它火锅
# 395       其他餐饮美食
# 433          湖北菜
# 458          自助餐
# 470         美食特产
# 587        香锅/烤鱼
# 679          台湾菜
# 682           闽菜
# 692           湘菜
# 762           熟食
# 818         其它烧烤
# 1213       上海本帮菜
