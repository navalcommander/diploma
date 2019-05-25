
# coding: utf-8

# In[22]:


import requests
from bs4 import BeautifulSoup as bs


# In[11]:


headers = {'accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8', 
           'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 YaBrowser/19.4.2.702 Yowser/2.5 Safari/537.36'}


# In[12]:


base_url = 'https://market.yandex.ru/product--smartfon-samsung-galaxy-s9-64gb/1968987605/reviews?hid=91491&page=6'


# In[62]:


def ire_parse(base_url, headers):
    session = requests.session()
    request = session.get(base_url, headers=headers)
    if request.status_code == 200:
        soup = bs(request.content,'html.parser')
        divs = soup.find_all('div', attrs = {'class': 'n-product-review-item i-bem n-product-review-item_collapsed_yes'})
        pluses = []
        for div in divs:
            pluses.append(div.find_all('dd', attrs = {'class': 'n-product-review-item__text'}))
            for i in range(len(pluses)):
                for j in range(len(pluses[i])):
                    print(pluses[i][j], end=' ')
                    
                print()
        print(pluses)
    else:
        print('ERROR')
ire_parse(base_url, headers)

