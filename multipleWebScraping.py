import urllib.request
from bs4 import BeautifulSoup

#URL TO SCRAP
wiki = "https://dlca.logcluster.org"

#Query the website and return the html to the variable 'page'
#For python3 use urllib.request.urlopen(wiki)
page = urllib.request.urlopen(wiki)

#Parse the html in hte 'page' variable, and store it in Beautiful Soup format
soup = BeautifulSoup(page, features='html.parser')
print('\n\nPage Scrapped !!! \n\n')
print(soup.title.string)

print('\n\nTITLE OF THE PAGE\n\n')
print(soup.title.string)

print('\n\nALL THE URLS IN THE WEB PAGE\n\n')
all_links = soup.find_all('a')

print('Total number of URLS = ', len(all_links))
print('\n\nLast 5 URLs in the page are: \n')

if len(all_links) < 5:
    last5 = all_links[len(all_links)-5:]
    for url in last5:
        print(url.get('href'))

emails = []

for url in all_links:
    if(str(url.get('href')).find('@')>0):
        emails.append(url.get('href'))
print('\n\nTotal Number of Email IDs Present: ', len(emails))

print('\n\nSome of the emails are: \n\n')
for email in emails[:5]:
    print(email)
