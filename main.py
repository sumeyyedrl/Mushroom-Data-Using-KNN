from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
from sklearn.preprocessing import LabelEncoder


mushroom = fetch_ucirepo(id=73)
type(mushroom)

x = mushroom.data.features
y = mushroom.data.targets
type(x)
type(y)

x.head()
y.head()

x.shape
x.describe().T()
x.isnull().sum()

x.drop("veil-type",axis=1,inplace=True)
x["stalk-root"].unique()
x["stalk-root"]=x["stalk-root"].fillna('n')

x=pd.get_dummies(x,columns=x.columns,drop_first=True)
x.head()
x.shape

correlation_matrix = x.corr()
plt.figure(figsize=(15, 9))
sns.heatmap(correlation_matrix)
plt.show(block=True)

y["poisonous"].unique()
y["poisonous"].value_counts().plot(kind="bar")
plt.show(block=True)

labelEncoder= LabelEncoder()
y["poisonous"]=labelEncoder.fit_transform(y["poisonous"])
y.head()

knn_model = KNeighborsClassifier()

knn_params = {"n_neighbors": range(2, 20)}
knn_gs_best = GridSearchCV(knn_model,knn_params,cv=3,n_jobs=-1,verbose=1).fit(x, y)
knn_gs_best.best_params_

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(x, y)

cv_results = cross_validate(knn_final,x,y,cv=10,scoring=["accuracy", "f1", "roc_auc", "recall", "precision"])

cv_results['test_accuracy']
cv_results['test_accuracy'].mean()

cv_results['test_f1']
cv_results['test_f1'].mean()

cv_results['test_roc_auc']
cv_results['test_roc_auc'].mean()

cv_results['test_recall']
cv_results['test_recall'].mean()

cv_results['test_precision']
cv_results['test_precision'].mean()
