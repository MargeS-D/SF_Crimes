#!/usr/bin/env python
# coding: utf-8

# In[75]:


import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon

import geopandas as gpd
from geopandas.tools import sjoin

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# In[76]:

df_police2018 = pd.read_csv("../FINAL PROJECT/DATA/2018_Police_Report.csv")
df_police2019 = pd.read_csv("../FINAL PROJECT/DATA/2019_Police_Report.csv")


# In[77]:


df_police2018.head()


# In[78]:


df_police2018.describe()


# In[79]:


df_police2018.info()


# In[80]:


df_police2019.info()


# In[81]:


df_police2018['geometry']=df_police2018.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)


# In[82]:


#Coordinate Reference System (CRS)

#WGS84 Latitude/Longitude: "EPSG:4326"

#2018

geo_police2018 = gpd.GeoDataFrame(df_police2018, geometry='geometry')
geo_police2018.crs= {'init':'epsg:4326'}


# In[83]:


san_fran=gpd.read_file('/Users/marietoudione/Documents/DATA MINING1/FINAL PROJECT/DATA/zipcodes.json')[['zip', 'geometry']]


# In[84]:


san_fran=gpd.GeoDataFrame(san_fran)


# In[85]:


san_fran.crs = {'init':'epsg:4326'}
san_fran = san_fran.set_geometry('geometry')
san_fran.head()


# In[86]:


type(geo_police2018.geometry[0])


# In[87]:


type(san_fran.geometry[0])


# In[88]:


df_police2018= gpd.tools.sjoin(geo_police2018, san_fran, how='left', op = 'within')


# In[89]:


df_police2018.head()


# In[90]:


#Count of crimes. day/hours/zip code/names

crime_time18 = df_police2018[['Incident Date','Incident Time','Incident Day of Week', 'geometry','zip']]


# In[91]:


crime_time18.loc[:, 'Date']= pd.to_datetime(crime_time18['Incident Date'])

crime_time18.loc[:, 'Hour']= pd.to_datetime(crime_time18['Incident Time'])
crime_time18.loc[:, 'Hour']= crime_time18.Hour.apply(lambda x: x.hour)


# In[92]:


crime_time18.head()


# In[93]:


crime_time18.plot(figsize=(10,5),marker='o', color='b', markersize=0.5)


# In[94]:


crime_time18_perday= crime_time18[['Incident Day of Week', 'zip', 'Hour']]


# In[95]:


#Create new variable crime

crime_time18_perday.loc[:, 'Crimes']=1


# In[96]:


crime_time18_perday = crime_time18_perday.groupby(['Incident Day of Week', 'zip', 'Hour']).sum().reset_index()

crime_time18_perday.sort_values('Crimes', ascending = False).head(10)


# In[97]:


sf_map = crime_time18_perday.merge(san_fran)
sf_map = gpd.GeoDataFrame(sf_map, geometry='geometry')
sf_map.crs = {'init': 'epsg:4326'}


# In[98]:


sf_map.plot(column='Crimes', cmap='Oranges', figsize=(20,10))


# In[99]:


#Dummy variables for 2018 dataset

crime_time18_perday= crime_time18_perday[['Crimes', 'Hour', 'Incident Day of Week', 'zip']]



dummy_18 = pd.get_dummies(crime_time18_perday)



X_18 = dummy_18.iloc[:, 1:]
Y_18 = dummy_18.iloc[:, 0]


#Random Forest
rnd_clf = RandomForestRegressor(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, oob_score=True)
rnd_clf.fit(X_18, Y_18)


rnd_clf.oob_score_


#2019

df_police2019['geometry']=df_police2019.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)





geo_police2019 = gpd.GeoDataFrame(df_police2019, geometry='geometry')
geo_police2019.crs= {'init':'epsg:4326'}




df_police2019 = gpd.tools.sjoin(geo_police2019, san_fran, how='left', op = 'within')


df_police2019.head()


crime_time19 = df_police2019[['Incident Date','Incident Time','Incident Day of Week', 'geometry','zip']]



crime_time19.loc[:, 'Hour']= pd.to_datetime(crime_time19['Incident Time'])
crime_time19.loc[:, 'Hour']= crime_time19.Hour.apply(lambda x: x.hour)




crime_time19.head()



crime_time19_perday= crime_time19[['Incident Day of Week', 'zip', 'Hour']]



#Create new variable crime

crime_time19_perday['Crimes']=1


crime_time19_perday = crime_time19_perday.groupby(['Incident Day of Week', 'zip', 'Hour']).sum().reset_index()



crime_time19_perday.sort_values('Crimes', ascending = False).head(10)


#Dummy variables

crime_time19_perday= crime_time19_perday[['Crimes', 'Hour', 'Incident Day of Week', 'zip']]


dummy_19 = pd.get_dummies(crime_time19_perday)


X_19 = dummy_19.iloc[:, 1:]
Y_19 = dummy_19.iloc[:, 0]


#Gradient Boosting

gbrt= GradientBoostingRegressor(max_depth=3, n_estimators=200, learning_rate=1.0)
gbrt.fit(X_18, Y_18)

gbrt.score(X_19, Y_19)


gbrt_pred= gbrt.predict(X_19)

crime_time19_perday['Pred_GBRT'] = pd.Series(gbrt_pred)



crime_time19_perday.head()



crime_time19_perday['Crimes']=crime_time19_perday['Crimes']/365

crime_time19_perday['Pred_GBRT']=crime_time19_perday['Pred_GBRT']/365



crime_time19_perday=np.round(crime_time19_perday, 2)


crime_time19_perday.head()



crime_time19_perday.to_json('../crimes_pred2019.json', orient='records', double_precision=2)





