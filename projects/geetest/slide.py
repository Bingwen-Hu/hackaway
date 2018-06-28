# http://www.bubuko.com/infodetail-2467118.html

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as Expect
from selenium.webdriver.support.wait import WebDriverWait

from PIL import Image




driver = webdriver.Firefox()
driver.get("https://account.geetest.com/login")

wait = WebDriverWait(driver, 10)

# email = driver.find_element_by_id('email')
# email.send_keys("test@163.com")

# buttom = wait.until
