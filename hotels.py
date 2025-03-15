import requests
from bs4 import BeautifulSoup
import pandas as pd
import openpyxl

#STEP 1: CRAWL
url = "https://www.goibibo.com/hotels/hotels-in-shimla-ct/"

headers = {'User-Agent': "Mozilla/5.0 (X11;Linux x8664) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 "
                         "Safari/537.36"}
response = requests.request("GET", url, headers=headers)

data = BeautifulSoup(response.text, 'html.parser')

# print(data)

#STEP 2: PARSE AND TRANSFORM
#Find all the sections with the specified class name
cards_data = data.find_all('div', attrs={'class': 'HotelCardV2styles__SRPCardInnerWrapper-sc-6przws-1 hblphz'})
#
# print('Total Number of Cards Found: ', len(cards_data))
#
# #Source code of hotel cards
# for card in cards_data:
#     print(card)

#Extract the hotel name and price per room
for card in cards_data:
    hotel_name = card.find('a')

    room_price = card.find('p', attrs={'class': 'HotelCardV2styles__OfferPrice-sc-6przws-18 FiTxV'})
    print(hotel_name.text, room_price.text)

#STEP 3: STORE DATA
scraped_data = []

for card in cards_data:
    card_details = {}
    hotel_name = card.find('a')

    room_price = card.find('p', attrs={'class': 'HotelCardV2styles__OfferPrice-sc-6przws-18 FiTxV'})
    card_details['hotel_name'] = hotel_name.text
    card_details['room_price'] = room_price.text

    scraped_data.append(card_details)
for card_details in scraped_data:
    print(card_details)

dataframe = pd.DataFrame(scraped_data)

dataframe.to_excel('hotels.xlsx', index=False)
