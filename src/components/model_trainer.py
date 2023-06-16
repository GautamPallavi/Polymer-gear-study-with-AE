import os
import sklearn.model_selection
import sklearn.ensemble
import sklearn as sns
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import warnings
import random as rnd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras. layers import Dense
from tensorflow.keras.layers import LeakyReLU, PReLU, ReLU,ELU
from tensorflow.keras.layers import Dropout
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import matplotlib_inline

directoryPath = os.getcwd()+'/data/AE/polymer_polymer/AEfeatures/features.csv'
df = pd.read_csv(directoryPath, header=None)
df=df.sample(frac=1)

#dictionary = {'polymer_polymer_healthy': 1, 'polymer_polymer_wear_1': 2, 'polymer_polymer_wear_2': 3}

#print(df[0])

"""
df['experiment_type_ordinal'] = df[0].map(dictionary)
df = df.drop(0, axis=1)
x=df.drop('experiment_type_ordinal',axis=1)
y=df['experiment_type_ordinal']

"""
x= df.drop([0],axis=1)
Y1= df[0]
encoder= LabelEncoder()
y1=encoder.fit_transform(Y1)
y=pd.get_dummies(y1).values

"""
#print(df.head())

print(x)
"""

#print(y)
#print(Y1)
#print(y1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=50)
RS= RobustScaler()
x_train=RS.fit_transform(x_train)
x_test=RS.transform(x_test)

rf_classifier = RandomForestClassifier(n_estimators=10,max_depth=100,max_leaf_nodes=10).fit(x_train,y_train)
rfprediction = rf_classifier.predict(x_test)

y_test_class=np.argmax(y_test,axis=1)
rfprediction_class=np.argmax(rfprediction,axis=1)
print(confusion_matrix(y_test_class,rfprediction_class))
print(accuracy_score(y_test_class,rfprediction_class))
print(classification_report(y_test_class,rfprediction_class))


"""
print(confusion_matrix(y_test,rfprediction))
print(accuracy_score(y_test,rfprediction))
print(classification_report(y_test,rfprediction))
"""
#--- ANN

RS= RobustScaler()
x_train=RS.fit_transform(x_train)
x_test=RS.transform(x_test)

"""

#sc= StandardScaler()
#x_train=sc.fit_transform(x_train)
#x_test=sc.transform(x_test)
#print(x_train)
#print(x_test)

"""
classifier=Sequential()

classifier.add(Dense(12,input_shape=(12,),activation='relu'))
classifier.add(Dense(6,input_shape=(12,),activation='relu'))
classifier.add(Dense(5,input_shape=(6,),activation='relu'))
classifier.add(Dense(4,input_shape=(5,),activation='relu'))
classifier.add(Dense(3,activation='softmax'))

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'] )

early_stopping=tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.01,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)


model_history=classifier.fit(x_train,y_train,validation_data=(x_test,y_test),
                             batch_size=10,epochs=50,callbacks=early_stopping)
prediction=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


"""
print(confusion_matrix(y_test,prediction))
print(accuracy_score(y_test,prediction))
print(classification_report(y_test,prediction))
"""
y_test_class=np.argmax(y_test,axis=1)
prediction_class=np.argmax(prediction,axis=1)
print(confusion_matrix(y_test_class,prediction_class))
print(accuracy_score(y_test_class,prediction_class))
print(classification_report(y_test_class,prediction_class))


#<------SVM

from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import uniform

x_train,x_test,y1_train,y1_test = train_test_split(x,y1,test_size=0.20,random_state=50)
RS= RobustScaler()
x_train=RS.fit_transform(x_train)
x_test=RS.transform(x_test)
svm = SVC(C=5,kernel="rbf",gamma=2) 
svm.fit(x_train,y1_train)

prediction=svm.predict(x_test)
print(confusion_matrix(y1_test,prediction))
print(accuracy_score(y1_test,prediction))
print(classification_report(y1_test,prediction))

