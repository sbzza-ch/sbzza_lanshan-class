import requests
import time

cookies={

}

url=''
from selenium.webdriver.common.by import By
from selenium import webdriver

while True:
    resp = requests.get(url,cookies=cookies)
    confirm_button = driver.find_element(By.XPATH, '//button[text()="确定"]')
    confirm_button.click()
    time.sleep(0.3)
    print(resp.text)