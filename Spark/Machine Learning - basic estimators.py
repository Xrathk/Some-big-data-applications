# Project 1 : Erotima 5
# importing libraries
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import os

# loading csv
os.chdir(r'Desktop\ΜΕΤΑΠΤΥΧΙΑΚΟ\Μεταπτυχιακό ΕΚΠΑ Τηλεπικοινωνίες\\2ο Εξάμηνο\Διαχείρηση Μεγάλων Δεδομένων\Big Data Projects\Project 1')

df = pd.read_csv(r'fake_job_postings.csv')

# saving column names
labels = list(df.columns)


## Naive bayes classifier
x = df[['telecommuting']]
y = df['fraudulent']
model = GaussianNB()
model.fit(x,y)
y_pred = model.predict(x)
# estimating performance
print("\nNaive bayes estimator Performance:\n")
precision = precision_score(y,y_pred) # precision
print("Precision = ",precision)
recall = recall_score(y,y_pred)
print("Recall = ",recall)
F1 = f1_score(y,y_pred) # F1 score
print("F1 score = ",F1)
print("\n-----------------------------------------------------------------------------------\n")

# new feature list - training set
features = df[['telecommuting','has_company_logo','has_questions']]
# preprocessing
y = df['fraudulent']
x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.25, random_state = 42)

## kNN classifier - 10 Neighbors
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(x_train,y_train)
y_pred2 = knn.predict(x_test)
# estimating performance
print("kNN nearest neighbors estimator Performance (10 neighbors):\n")
precision2 = precision_score(y_test,y_pred2, average='weighted', labels=np.unique(y_pred2)) # precision
print("Precision = ",precision2)
recall2 = recall_score(y_test,y_pred2, average='weighted' )
print("Recall = ",recall2)
F1_2 = f1_score(y_test,y_pred2, average='weighted') # F1 score
print("F1 score = ",F1_2,"\n")
print("\n-----------------------------------------------------------------------------------\n")

## SVM classifier - sigmoid kernel
svmc = svm.SVC(kernel='sigmoid') 
svmc.fit(x_train, y_train)
y_pred3 = svmc.predict(x_test)
# estimating performance
print("SVM sigmoid kernel estimator performance:\n")
precision3 = precision_score(y_test,y_pred3, average='weighted', labels=np.unique(y_pred3)) # precision
print("Precision = ",precision3)
recall3 = recall_score(y_test,y_pred3, average='weighted' )
print("Recall = ",recall3)
F1_3 = f1_score(y_test,y_pred3, average='weighted') # F1 score
print("F1 score = ",F1_3,"\n")
