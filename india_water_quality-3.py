#!/usr/bin/env python
# coding: utf-8

# In[417]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

##import os
##captureprint(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[418]:


data=pd.read_csv('water_dataX.csv',encoding="ISO-8859-1")
#data.fillna(0, inplace=True)
data.head()


# In[419]:


# let's check whether our dataset have any null/missing values
data.isnull().sum()


# In[420]:


data.dtypes


# In[421]:


#conversions
data['Temp']=pd.to_numeric(data['Temp'],errors='coerce')
data['D.O. (mg/l)']=pd.to_numeric(data['D.O. (mg/l)'],errors='coerce')
data['PH']=pd.to_numeric(data['PH'],errors='coerce')
data['B.O.D. (mg/l)']=pd.to_numeric(data['B.O.D. (mg/l)'],errors='coerce')
data['CONDUCTIVITY (µmhos/cm)']=pd.to_numeric(data['CONDUCTIVITY (µmhos/cm)'],errors='coerce')
data['NITRATENAN N+ NITRITENANN (mg/l)']=pd.to_numeric(data['NITRATENAN N+ NITRITENANN (mg/l)'],errors='coerce')
data['FECAL COLIFORM (MPN/100ml)']=pd.to_numeric(data['FECAL COLIFORM (MPN/100ml)'],errors='coerce')
data['TOTAL COLIFORM (MPN/100ml)Mean']=pd.to_numeric(data['TOTAL COLIFORM (MPN/100ml)Mean'],errors='coerce')
data.dtypes


# In[422]:


#initialization
start=2
end=1779
station=data.iloc [start:end ,0]
location=data.iloc [start:end ,1]
state=data.iloc [start:end ,2]
Temp=data.iloc [start:end ,3]
do= data.iloc [start:end ,4].astype(np.float64)
value=0
ph = data.iloc[ start:end,5]  
co = data.iloc [start:end ,6].astype(np.float64)   
  
year=data.iloc[start:end,11]
tc=data.iloc [2:end ,10].astype(np.float64)


bod = data.iloc [start:end ,7].astype(np.float64)
na= data.iloc [start:end ,8].astype(np.float64)
na.dtype


# In[423]:


data.head()


# In[424]:


data.shape


# In[425]:


data.dtypes


# In[426]:


data=pd.concat([station,location,state,Temp,do,ph,co,bod,na,tc,year],axis=1)
data. columns = ['station','location','state','Temp','do','ph','co','bod','na','tc','year']


# In[427]:


data.head(2)


# In[428]:


#calulation of Ph
data['npH']=data.ph.apply(lambda x: (100 if (8.5>=x>=7)  
                                 else(80 if  (8.6>=x>=8.5) or (6.9>=x>=6.8) 
                                      else(60 if (8.8>=x>=8.6) or (6.8>=x>=6.7) 
                                          else(40 if (9>=x>=8.8) or (6.7>=x>=6.5)
                                              else 0)))))


# In[429]:


#calculation of dissolved oxygen
data['ndo']=data.do.apply(lambda x:(100 if (x>=6)  
                                 else(80 if  (6>=x>=5.1) 
                                      else(60 if (5>=x>=4.1)
                                          else(40 if (4>=x>=3) 
                                              else 0)))))


# In[430]:


#calculation of total coliform
data['nco']=data.tc.apply(lambda x:(100 if (5>=x>=0)  
                                 else(80 if  (50>=x>=5) 
                                      else(60 if (500>=x>=50)
                                          else(40 if (10000>=x>=500) 
                                              else 0)))))


# In[431]:


#calc of B.D.O
data['nbdo']=data.bod.apply(lambda x:(100 if (3>=x>=0)  
                                 else(80 if  (6>=x>=3) 
                                      else(60 if (80>=x>=6)
                                          else(40 if (125>=x>=80) 
                                              else 0)))))


# In[432]:


#calculation of electrical conductivity
data['nec']=data.co.apply(lambda x:(100 if (75>=x>=0)  
                                 else(80 if  (150>=x>=75) 
                                      else(60 if (225>=x>=150)
                                          else(40 if (300>=x>=225) 
                                              else 0)))))


# In[433]:


#Calulation of nitrate
data['nna']=data.na.apply(lambda x:(100 if (20>=x>=0)  
                                 else(80 if  (50>=x>=20) 
                                      else(60 if (100>=x>=50)
                                          else(40 if (200>=x>=100) 
                                              else 0)))))

data.head()
data.dtypes


# In[434]:



data['wph']=data.npH * 0.165
data['wdo']=data.ndo * 0.281
data['wbdo']=data.nbdo * 0.234
data['wec']=data.nec* 0.009
data['wna']=data.nna * 0.028
data['wco']=data.nco * 0.281
data['wqi']=data.wph+data.wdo+data.wbdo+data.wec+data.wna+data.wco 
data


# In[435]:


data_new=data


# In[436]:


data_new.dtypes


# In[437]:


data_new.head(2)


# In[438]:


data_new=data_new.dropna()


# In[439]:


x1=data_new.drop(['location','state','wqi','station','year','npH','ndo','nco','nbdo','nec','nna','wph','wdo','wbdo','wec','wna','wco','na','tc'],axis=1)
y1=data_new['wqi']


# In[440]:


x1.head()


# In[441]:


plt.scatter(data_new['Temp'], data_new['wqi'])
plt.xlabel('Temperature')
plt.ylabel('water quality index')


# In[442]:


plt.scatter(data_new['ph'], data_new['wqi'])
plt.xlabel('Ph')
plt.ylabel('water quality index')


# In[443]:


plt.scatter(data_new['co'], data_new['wqi'])
plt.xlabel('conductivity')
plt.ylabel('water quality index')


# In[444]:


plt.scatter(data_new['bod'], data_new['wqi'])
plt.xlabel('biological oxygen demand')
plt.ylabel('water quality index')


# In[445]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# In[446]:


x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.3,random_state=100)


# In[447]:


y1_train.head(2)


# In[448]:


data_new.isnull().sum()


# In[449]:


data_new=data_new.dropna()


# In[450]:


reg=linear_model.LinearRegression()


# In[451]:


reg.fit(x1_train,y1_train)


# In[452]:


x1_train.head(2)


# In[453]:


reg.score(x1_train,y1_train)


# In[454]:


reg.score(x1_test,y1_test)


# In[455]:


reg.coef_


# In[456]:


regr1 = LinearRegression()


# In[457]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=6)))
features.append(('Minmaxscalor',MinMaxScaler()))
feature_union = FeatureUnion(features)
# create pipeline
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('linear', LinearRegression()))
model = Pipeline(estimators)
# evaluate pipeline
seed = 7
kfold = KFold(n_splits=10)
results = cross_val_score(regr1, x1, y1, cv=kfold)
print(results.mean())


# In[458]:


regr1 = LinearRegression()
regr1.fit(x1_train,y1_train)
regr1.score(x1_test,y1_test)


# In[459]:


x1_train.head(2)


# In[460]:


regr1.predict([[1,2,3,1,2]])


# In[461]:


## Polynomial regression


# In[462]:


from sklearn.preprocessing import PolynomialFeatures


# In[463]:


# from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(x1)
pol_reg = linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X_poly, y1, test_size=0.30 , random_state=1)
pol_reg.fit(X_train, y_train)
y_poly_pred=pol_reg.predict(X_test)
#plt.scatter(y,y_poly_pred)


# In[464]:


y_poly_pred2=pol_reg.predict(X_train)


# In[465]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[466]:


##Accuracy of test data using polynomial regression
rmse = np.sqrt(mean_squared_error(y_test,y_poly_pred))
r2 = r2_score(y_test,y_poly_pred)
print(rmse)
print(r2)


# In[ ]:


##Random forest regressor


# In[501]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x1_train, y1_train)


# In[509]:


y_pred = regressor.predict(x1_test)
y_train = regressor.predict(x1_train)
#np.set_printoptions(precision=2)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y1_test),1)),1))


# In[513]:


from sklearn.metrics import r2_score
r2_score(y_train,y1_train)


# In[505]:


from sklearn.metrics import r2_score
r2_score(y1_test, y_pred)


# In[467]:


## Decision tree regressor


# In[468]:


from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
num_folds = 18
seed = 77
kfold = KFold(n_splits=num_folds, random_state=seed,shuffle=True)
dt_model = DecisionTreeRegressor()
dtr=DecisionTreeRegressor()
results1 = cross_val_score(dt_model,x1, y1, cv=kfold)
accuracy=np.mean(abs(results1))
print('Average accuracy: ',accuracy)
print('Standard Deviation: ',results1.std())


# In[469]:


from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[470]:


reg_model2 = AdaBoostRegressor(n_estimators=10, base_estimator=dtr,learning_rate=1) 


# In[479]:


ab=reg_model2.fit(x1_train,y1_train)


# In[480]:


c=ab.score(x1_test,y1_test) ## For ada boost accuracy of x_test and y_test
c


# In[481]:


ab.score(x1_train,y1_train) ## For ada boost accuracy of x_train and y_train


# In[483]:


from sklearn import metrics
print('R2 Value:',metrics.r2_score(y1_train,ab.predict(x1_train)))


# In[486]:


model=GradientBoostingRegressor() ## Gradient boost regressor
model.fit(x1_train, y1_train)


# In[487]:


k=model.score(x1_test,y1_test)
k
## For gradient boost accuracy of x_test and y_test


# In[488]:


x1_train.head(4)


# In[491]:


model.predict([[4,3.25,3,1,2]])


# In[492]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# In[493]:


# build a classifier
clf = GradientBoostingRegressor()
# specify parameters and distributions to sample from
param_dist = {"criterion":["mse"],"min_samples_split":[10,20,40],"loss": ['ls'],"max_depth":[2,6,8],
             "min_samples_leaf":[20,40,100],"max_leaf_nodes":[5,20,100]}
samples = 5  # number of random samples 
randomCV = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=samples) #default cv = 3


# In[494]:


clf.get_params().keys()


# In[495]:


randomCV.fit(X_train,y_train)


# In[496]:


k1=randomCV.best_score_
print("R-Squared::{}".format(randomCV.best_score_))
print("Best Hyperparameters::\n{}".format(randomCV.best_params_))


# In[515]:


##Random forest regressor


# In[516]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x1_train, y1_train)


# In[517]:


y_pred = regressor.predict(x1_test)
y_train = regressor.predict(x1_train)
#np.set_printoptions(precision=2)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y1_test),1)),1))


# In[518]:


from sklearn.metrics import r2_score
r2_score(y_train,y1_train)


# In[519]:


from sklearn.metrics import r2_score
k1=r2_score(y1_test, y_pred)


# In[522]:


a_disp=pd.DataFrame({"parameters":['accuracy'],"Linear Regression":[results.mean()],"Polynomial Regression":[r2],"Decision_Tree":[accuracy],"adaboost":[c],"Gradient_boost":[k],"Random forest":[k1]})


# In[523]:


a_disp


# In[526]:


a_disp1=pd.DataFrame({"parameters":['Linear Regression','Polynomial Regression','Decision_Tree','adaboost','Gradient_boost','Random forest'],"Accuracy Values":[results.mean(),r2,accuracy,c,k,k1]})


# In[527]:


a_disp1


# In[528]:


get_ipython().run_cell_magic('HTML', '', '<style type="text/css">\n    table.dataframe td, table.dataframe th {\n        border-style: solid;\n    }\n</style>')


# In[ ]:

pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6, 7, 8]]))

#regressor here is the final model

