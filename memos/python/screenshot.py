from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as Expect
from selenium.webdriver.support.wait import WebDriverWait

from PIL import Image

driver = webdriver.Firefox()
driver.get("http://www.weather.com.cn/weather1d/101280101.shtml#search")
wait = WebDriverWait(driver, timeout=10)

element_class_name = 'sk01'
weather = wait.until(Expect.presence_of_element_located([By.CLASS_NAME, element_class_name]))
location = weather.location
size = weather.size
top = location['y']
bottom = location['y'] + size['height']
left = location['x']
right = location['x'] + size['width']
driver.save_screenshot('screenshot.png')

screen_img = Image.open('screenshot.png')
weather_img = screen_img.crop([left, top, right, bottom])
weather_img.save('weather.png')
driver.close()