from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector as selector
import bisect
import numpy as np

def position(columns, data):
    position = []
    c = data[columns]
    for i in c:
        position.append(float(float(i[0:2]) + float(i[3])))
    data.loc[:,columns] = position
    
def get_mode_means(data):
    m = {}
    for i in data:
        if data[i].dtype == 'object':
            value=data[i].mode()
        else:
            value=data[i].mean()
        m[i] = value
    return m
def dropColumns(columns,data):
    for i in columns:
        data.drop(labels=i, axis=1, inplace=True)
        
def date_toInt(data,columnName):
    if data[columnName].dtype == 'object':
        temp = list(data[columnName])
        birthDate = []

        for i in range(len(temp)):
            temp[i] += ' '
            bd = temp[i][-5:-1]
            birthDate.append(int(bd))
        data.loc[:,columnName] = birthDate
        
        
        
#          
def Feature_Encoder(X,cols):
    enc = {}
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
        if type(lbl.classes_) != list:
            le_classes = lbl.classes_.tolist()
        else:
            le_classes = lbl.classes_
      
        bisect.insort_left(le_classes, 'other')
        lbl.classes_ = le_classes
        lbl.classes_ = np.array(lbl.classes_)
        enc[c] = lbl
    return  X,enc
def encoder_transform(X,cols,encoder):
   
    for c in cols:
        le = encoder[c]
        X[c] = X[c].map(lambda s: 'other' if s not in le.classes_ else s)
        X[c] = le.transform(list(X[c].values))
        
    return X
def featureScaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X
def Preprocessing_Scaling(X_train, X_test):
    mx = StandardScaler()
    numerical_columns_selector = selector(dtype_exclude=object)
    col = numerical_columns_selector(X_train)
    X_train_scaled=mx.fit_transform(X_train[col])
    col = numerical_columns_selector(X_test)
    X_test_scaled=mx.transform(X_test[col])
    return X_train_scaled, X_test_scaled , mx
def Preprocessing_ScalingX(X):
    mx = MinMaxScaler()
    numerical_columns_selector = selector(dtype_exclude=object)
    col = numerical_columns_selector(X)
    X_train_scaled=mx.fit_transform(X[col])
    return X_train_scaled
