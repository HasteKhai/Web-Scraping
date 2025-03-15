import requests
from bs4 import BeautifulSoup

url = "https://www.goibibo.com/hotels/hotels-in-shimla-ct/"

headers = {'User-Agent': "Mozilla/5.0 (X11;Linux x8664) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 "
                         "Safari/537.36"}
response = requests.request("GET", url, headers=headers)

data = BeautifulSoup(response.text, 'html.parser')

#Find all images
images = data.find_all('img', src=True)

print('Number of images:', len(images))

for image in images:
    print(image)

#Select src tag
image_src = [x['src'] for x in images]

#Select only jp format images
image_src = [x for x in image_src if x.endswith('.jpg')]

for image in image_src:
    print(image)

image_count = 1
MAX_IMAGES = 10

for image in image_src:
    if image_count == MAX_IMAGES:
        break
    with open('image_'+str(image_count) + '.jpg', 'wb') as f:
        res = requests.get(image)
        f.write(res.content)
    image_count = image_count+1
