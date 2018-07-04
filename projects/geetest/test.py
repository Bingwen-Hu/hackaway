from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as Expect
from selenium.webdriver.support.wait import WebDriverWait

from PIL import Image
import time
import requests
import json
import numpy as np
from skimage.morphology import erosion

def get_snap():
    radar = wait.until(Expect.presence_of_element_located((By.CLASS_NAME, 'geetest_radar_tip')))
    radar.click()
    time.sleep(1.5)
    org = wait.until(Expect.presence_of_element_located((By.CLASS_NAME, 'geetest_canvas_img')))
    time.sleep(1)
    location = org.location
    size = org.size
    top = location['y']
    bottom = location['y'] + size['height']
    left = location['x']
    right = location['x'] + size['width']
    driver.save_screenshot('./org.png')
    org = Image.open('./org.png')
    org_crop = org.crop((left, top, right, bottom))
    org_crop.save("./org_crop.png")

    slider = wait.until(Expect.presence_of_element_located((By.CLASS_NAME, 'geetest_slider_button')))
    slider.click()
    time.sleep(1.5)
    new = wait.until(Expect.presence_of_element_located((By.CLASS_NAME, 'geetest_canvas_img')))
    time.sleep(1)
    location = new.location
    size = new.size
    top = location['y']
    bottom = location['y'] + size['height']
    left = location['x']
    right = location['x'] + size['width']
    driver.save_screenshot('./new.png')
    new = Image.open('./new.png')
    new_crop = new.crop((left, top, right, bottom))
    new_crop.save("./new_crop.png")



def interface():
    org = "org_crop.png"
    new = "new_crop.png"
    
    files = {
        'old': (org, open(org, 'rb'), 'image/png'),
        'new': (new, open(new, 'rb'), 'image/png'),
    }
    r = requests.post('http://119.84.122.135:27701/slider', files=files)
    res = json.loads(r.text)
    return res['distance']


if __name__ == '__main__':
    
    driver = webdriver.Firefox()
    driver.get("https://account.geetest.com/login")

    wait = WebDriverWait(driver, 10)

    email = driver.find_element_by_id('email')
    email.send_keys("test@163.com")
    get_snap()
    start = time.time()
    move = interface()
    end = time.time()
    print(end - start)
    steps = [move/2, move-move/2]
    slider = wait.until(Expect.presence_of_element_located((By.CLASS_NAME, 'geetest_slider_button')))
    ActionChains(driver).click_and_hold(slider).perform()
    for step in steps:
        ActionChains(driver).move_by_offset(xoffset=step, yoffset=0).perform()
        ActionChains(driver).move_by_offset(xoffset=3, yoffset=0).perform()
        time.sleep(0.4)
    else:
        ActionChains(driver).move_by_offset(xoffset=-3, yoffset=0).perform()
        ActionChains(driver).move_by_offset(xoffset=-3, yoffset=0).perform()

    ActionChains(driver).release().perform()

    time.sleep(3)
    driver.close()