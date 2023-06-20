import os
import sklearn.model_selection
import sklearn.ensemble
import sklearn as sns
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
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
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import RandomizedSearchCV
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import matplotlib_inline
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint


directoryPath = os.getcwd()+'/data/AE/polymer_polymer/AEfeatures/features.csv'
df = pd.read_csv(directoryPath, header=None)
df=df.sample(frac=1).reset_index(drop=True)

x= df.drop([0],axis=1)
Y1= df[0]
encoder= LabelEncoder()
y1=encoder.fit_transform(Y1)
y=pd.get_dummies(y1).values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=50)
RS= RobustScaler()
x_train=RS.fit_transform(x_train)
x_test=RS.transform(x_test)

#print(x_train.shape)
#print(x_train)
#print(y_train.shape[1:])

def build_model(n_hidden = 5, n_neurons=32, learning_rate=3e-3, input_shape=x_train.shape[1:]):
  
  '''
  Builds a keras ANN for Multi-Class Classification i.e. output classes which are mutually exclusive
  '''


  model = Sequential()
  options = {"input_shape": input_shape}

  # Adding input and hidden layers
  for layer in range(n_hidden):
    model.add(Dense(n_neurons,activation="relu",**options))
    options = {}

  # Adding output layer having 3 neurons, 1 per class
  model.add(Dense(3,activation='softmax'))

  # Creating instance of adam optimizer
  opt = Adam(learning_rate=learning_rate)
  model.compile(optimizer=opt,loss='categorical_crossentropy',metrics='accuracy')

  return model

keras_cls = KerasClassifier(build_model)

param_dict = {
    "n_hidden" : (2,3),
    "n_neurons" : tuple(range(2,7)),
    "learning_rate" : (3e-2,3e-3,3e-4)
}

model_cv = RandomizedSearchCV(keras_cls,param_dict, n_iter=10, cv=5,
                              n_jobs=-1)


model_cv.fit(
    x_train, y_train, epochs=1000,
    #validation_data = 0.2,
    validation_data = (x_test,y_test),
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=0, patience=10,mode="min",min_delta=0.01)],
    batch_size=10,
    verbose=0 
)

print(model_cv.best_params_)

print(model_cv.best_score_)

best_set = model_cv.best_params_

param_grid = {
    'n_hidden' : [model_cv.best_params_['n_hidden']],
                  
    'n_neurons': [model_cv.best_params_['n_neurons']],

    "learning_rate" : list([model_cv.best_params_['learning_rate']])}
                       


grid_search=GridSearchCV(estimator=model_cv,param_grid=param_dict,cv=10,n_jobs=-1,verbose=2)
#GridSearchCV()
grid_search.fit(x_train,y_train)
print(grid_search.best_estimator_)

print(param_dict)
pred_classes = model_cv.predict(x_test)
y_test_classes = np.argmax(y_test,axis=1)
print(classification_report(y_test_classes,pred_classes),"\n\n")
print(confusion_matrix(y_test_classes,pred_classes))

pd.DataFrame(model_cv.history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

"""
history_dict = model_cv.history

# learning curve
# accuracy
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

# loss
loss = history_dict['loss']
val_loss = history_dict['val_loss']

# range of X (no. of epochs)
epochs = range(1, len(acc) + 1)

# plot
# "r" is for "solid red line"
plt.plot(epochs, acc, 'r', label='Training accuracy')
# b is for "solid blue line"
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""

#####  RANDOM FOREST

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 1000,10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,14]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,6,8]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}
print(random_grid)

rf=RandomForestClassifier()
rf_randomcv=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=100,cv=3,verbose=2,random_state=100,n_jobs=-1)


### fit the randomized model
rf_randomcv.fit(x_train,y_train)

print(rf_randomcv.best_params_)

best_random_grid=rf_randomcv.best_estimator_

from sklearn.metrics import accuracy_score
prediction=best_random_grid.predict(x_test)
y_test_class=np.argmax(y_test,axis=1)
prediction_class=np.argmax(prediction,axis=1)
print(confusion_matrix(y_test_class,prediction_class))
print(accuracy_score(y_test_class,prediction_class))
print(classification_report(y_test_class,prediction_class))

param_grid = {
    'criterion': [rf_randomcv.best_params_['criterion']],
    'max_depth': [rf_randomcv.best_params_['max_depth']],
    'max_features': [rf_randomcv.best_params_['max_features']],
    'min_samples_leaf': [rf_randomcv.best_params_['min_samples_leaf'], 
                         rf_randomcv.best_params_['min_samples_leaf']+2, 
                         rf_randomcv.best_params_['min_samples_leaf'] + 4],
    'min_samples_split': [rf_randomcv.best_params_['min_samples_split'] - 2,
                          rf_randomcv.best_params_['min_samples_split'] - 1,
                          rf_randomcv.best_params_['min_samples_split'], 
                          rf_randomcv.best_params_['min_samples_split'] +1,
                          rf_randomcv.best_params_['min_samples_split'] + 2],
    'n_estimators': [rf_randomcv.best_params_['n_estimators'] - 200, rf_randomcv.best_params_['n_estimators'] - 100, 
                     rf_randomcv.best_params_['n_estimators'], 
                     rf_randomcv.best_params_['n_estimators'] + 100, rf_randomcv.best_params_['n_estimators'] + 200]
}

print(param_grid)

rf=RandomForestClassifier()
grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=10,n_jobs=-1,verbose=2)
grid_search.fit(x_train,y_train)
print(grid_search.best_estimator_)
best_grid=grid_search.best_estimator_

y_pred=best_grid.predict(x_test)

y_test_class=np.argmax(y_test,axis=1)
prediction_class=np.argmax(y_pred,axis=1)
print(confusion_matrix(y_test_class,prediction_class))
print(accuracy_score(y_test_class,prediction_class))
print(classification_report(y_test_class,prediction_class))

#print(confusion_matrix(y_test,y_pred))
#print("Accuracy Score {}".format(accuracy_score(y_test,y_pred)))
#print("Classification report: {}".format(classification_report(y_test,y_pred)))


###SVM

from sklearn.svm import SVC

x_train,x_test,y1_train,y1_test = train_test_split(x,y1,test_size=0.20,random_state=50)
RS= RobustScaler()
x_train=RS.fit_transform(x_train)
x_test=RS.transform(x_test)

svm = SVC()
svm_param_grid = {'C': [0.1, 1, 10, 100],
               'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
               'gamma': ['scale', 'auto'],
               'degree': [2, 3, 4, 5],
               'coef0': [0.0, 0.1, 0.2, 0.5]}

random_search = RandomizedSearchCV(svm, param_distributions=svm_param_grid, n_iter=10, cv=5)
random_search.fit(x_train, y1_train)

best_params = [random_search.best_params_]
print("Best Hyperparameters:", best_params)

svm_prediction=random_search.predict(x_test)
#prediction=svm.predict(x_test)

print(confusion_matrix(y1_test,svm_prediction))
print(accuracy_score(y1_test,svm_prediction))
print(classification_report(y1_test,svm_prediction))


svm = SVC()
grid_search = GridSearchCV(estimator=svm, param_grid=svm_param_grid, cv=5)
grid_search.fit(x_train, y1_train)
gcv_best_params = grid_search.best_params_

print("Best Hyperparameters:", gcv_best_params)
svm_y_pred_gcv= grid_search.predict(x_test)
print(classification_report(y1_test, svm_y_pred_gcv))
print(confusion_matrix(y1_test,svm_prediction))
print(accuracy_score(y1_test,svm_prediction))
