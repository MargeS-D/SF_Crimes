#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import geopandas as gpd
from geopandas.tools import sjoin


# In[63]:


words= open(r"/Users/marietoudione/Documents/DATA MINING1/FINAL PROJECT/DATA/sf_wc1.csv").read()


# In[64]:


#df_police2010_2017.info()


# In[65]:


#df_filt= df_police2010_2017[['Category']]


# In[66]:


text = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and word != 'RT'
                            ])


# In[74]:


from matplotlib.pyplot import imread

sf_mask=np.array(Image.open('/Users/marietoudione/Documents/DATA MINING1/FINAL PROJECT/DATA/sf.jpg'))


# In[75]:


sf_mask


# In[76]:


wordcloud = WordCloud( 
                      
                      stopwords=STOPWORDS,
                      background_color='white',
                      width=1800,
                      height=1400,
                        max_words=1000,
                        contour_width=2,
                        contour_color='steelblue',
                      mask=sf_mask
            ).generate(text)


# In[77]:


plt.figure(figsize=[20,10])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('/Users/marietoudione/Documents/DATA MINING1/FINAL PROJECT/DATA/SF_wc1.png', dpi=300)
plt.show()


# In[ ]:




