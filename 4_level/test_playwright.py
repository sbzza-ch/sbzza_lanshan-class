import asyncio
import re
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser =await p.chromium.launch(
            channel="msedge",
                                    headless=False)
        context =await browser.new_context()
        page =await context.new_page()
        await page.goto("https://space.bilibili.com/1126128950?spm_id_from=333.1007.0.0")

        await page.get_by_role("button", name="关闭验证").click()
        await page.get_by_text("立即登录").click()
        await page.locator("div").filter(has_text=re.compile(r"^QQ登录$")).click()
        await page.locator("iframe[name=\"ptlogin_iframe\"]").content_frame.get_by_role("link",
                                                                                  name="概念神の情绪迭代").click()


        await context.close()
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())