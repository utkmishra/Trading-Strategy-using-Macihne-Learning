#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:45:58 2019

@authors: Habeeb Olawin, Alexander Ilnytsky, Utkarsh Mishra
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA
from sklearn.metrics import accuracy_score
from sklearn import metrics
from scipy import stats



data = pd.read_csv("/Users/h_olawin/Dropbox/University of Illinois/DataAnalytics/real data/SYF US.csv")

data.columns = ['PX_OPEN','PX_OFFICIAL_CLOSE','PX_VOLUME','EQY_SH_OUT', 'RSI_3D','RSI_14D','IVOL_MONEYNESS','EQY_INST_PCT_SH_OUT',
                'PCT_INSIDER_SHARES_OUT','PUT_CALL_VOLUME_RATIO_CUR_DAY','PUT_CALL_OPEN_INTEREST_RATIO','MF_BLCK_1D','MF_NONBLCK_1D','CMCI',
                'FEAR_GREED', 'MACD_DIFF','volume%','return_this_day','return_next_day', 'up_down']




list_important_features = [2,9]
data = data.dropna()


X = data.iloc[:,4:17].values
y = data.iloc[:,19]

data = data.dropna()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=False)
y_test

#%% scatter plot correlation
import matplotlib
import matplotlib.pyplot as plt

print("Exploratory data analysis of feature importance")
indicators = ['RSI_3D','RSI_14D','IVOL_MONEYNESS','EQY_INST_PCT_SH_OUT','PCT_INSIDER_SHARES_OUT','PUT_CALL_VOLUME_RATIO_CUR_DAY','PUT_CALL_OPEN_INTEREST_RATIO','MF_BLCK_1D','MF_NONBLCK_1D','CMCI',
                'FEAR_GREED', 'MACD_DIFF','volume%']

sns.pairplot(data[indicators], size=2.5)
plt.savefig('SYF.png')
plt.show()


#%% heat map correlation

cm = np.corrcoef(data[indicators].values.T)
#sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 6},
                 yticklabels=indicators,
                 xticklabels=indicators)

#plt.tight_layout()
plt.savefig('SYFheat.png')
plt.show()

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)



params = [2, 5, 10, 15, 20, 28]
for i in params:
    print('N_estimators: ', i )
    RF = RandomForestClassifier(criterion='gini', max_depth=4, n_estimators=i,random_state=42)
    RF.fit(X_train,y_train)
    scores = cross_val_score(estimator=RF, X=X_train, y=y_train, cv=10, n_jobs=1)
    print('in-sample accuray: ', np.mean(scores)) 


#%% Random forest feature importance
feat_labels = data.columns[4:17]
rf=RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(X_train,y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))

plt.title('Random forest feature importance based on Updown')
plt.bar(range(X_train.shape[1]),importances[indices],align = 'center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation = 90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

#%%
# PCA transformation for logistic regression.
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)



lr=LogisticRegression()
lr=lr.fit(X_train_std,y_train)



#pca = PCA(n_components=2)
pca = PCA(n_components=6)

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

print("Plotting graph for feature 1 vs feature 2")
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()


# Training logistic regression classifier.

lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)

y_train_pred=lr.predict(X_train_pca)
print("Logistic Regression on pca transformed training data accuracy n=10:")
print( metrics.accuracy_score(y_train, y_train_pred))



y_test_pred = lr.predict(X_test_pca)
print("Logistic Regression on pca transformed testing data accuracy n=10:")
print( metrics.accuracy_score(y_test, y_test_pred) )

returns = data.iloc[652:818,18]

y_pred_logistic = lr.predict(X_test_pca)



m_lr_after_trans = y_pred_logistic*returns
m_lr_after_trans 
total_m_lr_after_trans = sum(m_lr_after_trans)
print(total_m_lr_after_trans)


#plt.plot(m_lr_after_trans.cumsum())

# 3-day rule for logistic regression being applied
for i in range(len(returns)-3):
    if returns[i+653] < 0 and returns[i+654] < 0 and returns[i+655] < 0:
        y_pred_logistic[i+3] = 0
        
     

m_lr_after_trans = y_pred_logistic*returns
m_lr_after_trans 
total_m_lr_after_trans = sum(m_lr_after_trans)
print(total_m_lr_after_trans)


pca = PCA(n_components=8)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# Training logistic regression classifier.
lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)
y_test_pred = lr.predict(X_test_pca)
#print("Logistic Regression on pca transformed testing data accuracy n=10:")
#print( metrics.accuracy_score(y_test, y_test_pred) )

returns = data.iloc[652:818,18]

y_pred_logistic = lr.predict(X_test_pca)



m_lr_after_trans = y_pred_logistic*returns
m_lr_after_trans 
total_m_lr_after_trans = sum(m_lr_after_trans)
print(total_m_lr_after_trans)


#plt.plot(m_lr_after_trans.cumsum())


for i in range(len(returns)-3):
    if returns[i+653] < 0 and returns[i+654] < 0 and returns[i+655] < 0:
        y_pred_logistic[i+3] = 0
        
     

m_lr_after_trans = y_pred_logistic*returns
m_lr_after_trans 
total_m_lr_after_trans = sum(m_lr_after_trans)
print(total_m_lr_after_trans)


#plt.plot(m_lr_after_trans.cumsum())
#compare=returns
#compare
#compare.cumsum()
#print(compare.cumsum().tail(1))
#plt.plot(compare.cumsum())



pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
# Training logistic regression classifier.
lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)
#y_train_pred=lr.predict(X_train_std)
#print("Logistic Regression on pca transformed training data accuracy n=10:")
#print( metrics.accuracy_score(y_train, y_train_pred))
y_test_pred = lr.predict(X_test_pca)
#print("Logistic Regression on pca transformed testing data accuracy n=10:")
#print( metrics.accuracy_score(y_test, y_test_pred) )

returns = data.iloc[652:818,18]

y_pred_logistic = lr.predict(X_test_pca)



m_lr_after_trans = y_pred_logistic*returns
m_lr_after_trans 
total_m_lr_after_trans = sum(m_lr_after_trans)
print(total_m_lr_after_trans)



for i in range(len(returns)-3):
    if returns[i+653] < 0 and returns[i+654] < 0 and returns[i+655] < 0:
        y_pred_logistic[i+3] = 0
        
     

m_lr_after_trans = y_pred_logistic*returns
m_lr_after_trans 
total_m_lr_after_trans = sum(m_lr_after_trans)
print(total_m_lr_after_trans)



pca = PCA(n_components=12)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
# Training logistic regression classifier.
lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)
#y_train_pred=lr.predict(X_train_std)
#print("Logistic Regression on pca transformed training data accuracy n=10:")
#print( metrics.accuracy_score(y_train, y_train_pred))
y_test_pred = lr.predict(X_test_pca)
#print("Logistic Regression on pca transformed testing data accuracy n=10:")
#print( metrics.accuracy_score(y_test, y_test_pred) )

returns = data.iloc[652:818,18]

y_pred_logistic = lr.predict(X_test_pca)



m_lr_after_trans = y_pred_logistic*returns
m_lr_after_trans 
total_m_lr_after_trans = sum(m_lr_after_trans)
print(total_m_lr_after_trans)


#plt.plot(m_lr_after_trans.cumsum())


for i in range(len(returns)-3):
    if returns[i+653] < 0 and returns[i+654] < 0 and returns[i+655] < 0:
        y_pred_logistic[i+3] = 0
        
     

m_lr_after_trans = y_pred_logistic*returns
m_lr_after_trans 
total_m_lr_after_trans = sum(m_lr_after_trans)
print(total_m_lr_after_trans)


#%% PCA transformation for SVM
print("svm on pca transformed training data:")
for n in (6,8,10,12):

    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    svm=SVC(kernel='rbf')
    svm=svm.fit(X_train_pca,y_train)


    y_train_pred=svm.predict(X_train_pca)
    y_test_pred=svm.predict(X_test_pca)



    y_pred_svm = svm.predict(X_test_pca)
    m_svm_after_trans = y_pred_svm*returns
    m_svm_after_trans 
    m_svm_after_trans.cumsum()

   
    total_m_svm_after_trans = sum(m_svm_after_trans)
    print(total_m_svm_after_trans)

    for i in range(len(returns)-3):
        if returns[i+653] < 0 and returns[i+654] < 0 and returns[i+655] < 0:
            y_pred_svm[i+3] = 0
    
    m_pca_after_trans = y_pred_svm*returns
    m_pca_after_trans 
    total_m_pca_after_trans = sum(m_pca_after_trans)
    print(total_m_pca_after_trans)


#%%%
print("RF on PCA transformed training data:")
pca = PCA(n_components=10)

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy',
                                max_features =10,
                                oob_score = True,
                                max_depth = 3,
                                n_estimators=1000, 
                                random_state=0,
                                n_jobs=-1)

forest.fit(X_train_pca, y_train)

y_train_pred=forest.predict(X_train_pca)
y_test_pred=forest.predict(X_test_pca)

y_pred_forest = forest.predict(X_test_pca)

m_forest_after_trans = y_pred_forest*returns
m_forest_after_trans 
m_forest_after_trans.cumsum()
total_m_forest_after_trans = sum(m_forest_after_trans)
print(total_m_forest_after_trans)


for i in range(len(returns)-3):
    if returns[i+653] < 0 and returns[i+654] < 0 and returns[i+655] < 0:
        y_pred_forest[i+3] = 0
    
m_forest_after_trans = y_pred_forest*returns
m_forest_after_trans 
m_forest_after_trans.cumsum()
total_m_forest_after_trans = sum(m_forest_after_trans)
print(total_m_forest_after_trans)




forest = RandomForestClassifier(criterion='entropy',
                                max_features = 10,
                                oob_score = True,
                                max_depth = 3,
                                n_estimators=1000, 
                                random_state=0,
                                n_jobs=-1)
forest.fit(X_train_std, y_train)

y_train_pred=forest.predict(X_train_std)
y_test_pred=forest.predict(X_test_std)

y_pred_forest = forest.predict(X_test_std)

m_forest_after_trans = y_pred_forest*returns
m_forest_after_trans 
m_forest_after_trans.cumsum()
total_m_forest_after_trans = sum(m_forest_after_trans)
print(total_m_forest_after_trans)



for i in range(len(returns)-3):
    if returns[i+653] < 0 and returns[i+654] < 0 and returns[i+655] < 0:
        y_pred_forest[i+3] = 0
    
m_forest_after_trans = y_pred_forest*returns
m_forest_after_trans 
m_forest_after_trans.cumsum()
total_m_forest_after_trans = sum(m_forest_after_trans)
print(total_m_forest_after_trans)

#print('Train Accuracy: %.3f' % forest.score(X_train_std, y_train))
#print('Test Accuracy: %.3f' % forest.score(X_test_std, y_test))



pca = PCA(n_components=12)

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='gini',
                                max_features =4,
                                oob_score = True,
                                max_depth = 2,
                                n_estimators=1000, 
                                random_state=0,
                                n_jobs=-1)

forest.fit(X_train_pca, y_train)

y_train_pred=forest.predict(X_train_pca)
y_test_pred=forest.predict(X_test_pca)

y_pred_forest = forest.predict(X_test_pca)

m_forest_after_trans = y_pred_forest*returns
m_forest_after_trans 
m_forest_after_trans.cumsum()
total_m_forest_after_trans = sum(m_forest_after_trans)
print(total_m_forest_after_trans)


for i in range(len(returns)-3):
    if returns[i+653] < 0 and returns[i+654] < 0 and returns[i+655] < 0:
        y_pred_forest[i+3] = 0
    
m_forest_after_trans = y_pred_forest*returns
m_forest_after_trans 
m_forest_after_trans.cumsum()
total_m_forest_after_trans = sum(m_forest_after_trans)
print(total_m_forest_after_trans)
























