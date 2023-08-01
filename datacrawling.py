from lxml import html
import requests
import re
from urllib import request

res = requests.get('https://leagueoflegends.fandom.com/wiki/League_of_Legends:_Wild_Rift')
tree = html.fromstring(res.text)
urls = [re.sub(r"(^(?!png).+png)(.*)", r"\1", i) for i in tree.xpath('//div[@class="columntemplate"]/ul/li/span/span/a/img/@data-src')]
for url in urls:
    filename = url.split('/')[-1]
    print(f"{url} \n")
    print(f"filename: {filename}")
    response = request.urlretrieve(url, "crawled_images\\"+filename)



