# http://www.bubuko.com/infodetail-2467118.html

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as Expect
from selenium.webdriver.support.wait import WebDriverWait

from PIL import Image
import time
import numpy as np
from skimage.morphology import erosion

driver = webdriver.Firefox()
driver.get("https://account.geetest.com/login")

wait = WebDriverWait(driver, 10)

email = driver.find_element_by_id('email')
email.send_keys("test@163.com")



radar = wait.until(Expect.presence_of_element_located((By.CLASS_NAME, 'geetest_radar_tip')))
radar.click()
time.sleep(3)
driver.save_screenshot('./org.png')
slider = wait.until(Expect.presence_of_element_located((By.CLASS_NAME, 'geetest_slider_button')))
slider.click()
time.sleep(3)
driver.save_screenshot('./new.png')


def distance():
    org = Image.open("org.png").convert("L")
    new = Image.open("new.png").convert("L")
    org_np = np.array(org)
    new_np = np.array(new)
    diff = np.abs(new_np - org_np)
    ero = erosion(diff)
    img = Image.fromarray(ero)
    img.save("diff.png")



distance()
