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


def distance():
    org = Image.open("org_crop.png").convert("L")
    new = Image.open("new_crop.png").convert("L")
    org_np = np.array(org)
    new_np = np.array(new)
    diff = np.abs(new_np - org_np)
    ero = erosion(diff)
    ero = np.where(ero > 210, 0, ero)
    img = Image.fromarray(ero)
    img.save("diff.png")


get_snap()
distance()

def detect_v():
    img = Image.open('diff.png')
    data = np.array(img)
    x, y = data.shape
    print("%d rows, %d columns" % (x, y))
    vsums = np.sum(data, axis=0)
    result = []
    for i, s in enumerate(vsums):
        if s > 10:
            if i+1 == y:
                result.append(i)
            elif vsums[i+1] < 10:
                result.append(i)

    if len(result) == 2:
        return result[1] - result[0] + 5
    else:
        return result[0]/2


move = detect_v()


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
success = driver.find_element_by_class_name('geetest_success_radar_tip_content')
if success.text == '':
    get_snap()
    distance()
    move = detect_v()
    steps = [move/2, move-move/2]

    ActionChains(driver).click_and_hold(slider).perform()
    for step in steps:
        ActionChains(driver).move_by_offset(xoffset=step, yoffset=0).perform()
        ActionChains(driver).move_by_offset(xoffset=3, yoffset=0).perform()
        time.sleep(1.4)
    else:
        ActionChains(driver).move_by_offset(xoffset=-3, yoffset=0).perform()
        ActionChains(driver).move_by_offset(xoffset=-3, yoffset=0).perform()

    ActionChains(driver).release().perform()
    time.sleep(2)
    
driver.close()