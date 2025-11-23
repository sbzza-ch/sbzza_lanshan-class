import asyncio
from playwright.async_api import async_playwright

async def async_click_btn():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            channel="msedge")
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto( "https://www.bilibili.com/video/BV1J1ycB2EPv/?spm_id_from=333.1387.homepage.video_card.click&vd_source=669102571eb2e49655fc1d352a8bf358", timeout=30000)
        await page.wait_for_load_state("networkidle")
        await page.get_by_text("登录").nth(1).click()
        await page.wait_for_timeout(1000)

        await page.get_by_text("QQ登录").nth(0).click()

        await page.get_by_text("QQ手机版").wait_for(state="visible", timeout=30000)
        await page.get_by_text("QQ手机版").click()
#-------------------前方施工中 ----------------------------

        await asyncio.Future()
        while True:
            await asyncio.sleep(10000)
        await browser.close()
if __name__ == "__main__":
    asyncio.run(async_click_btn())
#下下位替代
'''import webbrowser
import easyocr
import pyautogui as pg
import os
webbrowser.open("https://www.bilibili.com/video/BV1J1ycB2EPv/?spm_id_from=333.1387.upload.video_card.click&vd_source=669102571eb2e49655fc1d352a8bf358")
reader = easyocr.Reader(['ch_sim','en'])

def find_text(find_str):
    pg.screenshot('shot.png')
    result=reader.readtext('shot.png')
    for r in result:
        if find_str in r[1]:
            lt=r[0][0]
            rb=r[0][2]
            center_pos=(lt[0]+rb[0])/2,(lt[1]+rb[1])/2
            os.remove('shot.png')
            return center_pos
    os.remove('shot.png')
    return None
if __name__=='__main__':
    center_pos=find_text('收藏')
    pg.moveTo(center_pos)
    pg.click()
    center_pos=find_text('默认收藏夹')
    pg.moveTo(center_pos)
    pg.click()
    center_pos = find_text('确定')
    pg.moveTo(center_pos)
    pg.click()'''














