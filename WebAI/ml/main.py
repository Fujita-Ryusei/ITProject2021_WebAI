
import pandas as pd
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
from collections import defaultdict

def choose_model(x_train,x_test,y_train,y_test,model):
        if model == "RandomForestClassifier":
                return RFC(x_train,x_test,y_train,y_test)
        elif model == "RandomForestRegressor":
                return RFR(x_train,x_test,y_train,y_test)

def RFC(x_train,x_test,y_train,y_test):
        model = RandomForestClassifier()
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        return accuracy_score(y_test,y_pred)

def RFR(x_train,x_test,y_train,y_test):
        #model = LogisticRegression()
        model = RandomForestRegressor(max_depth = 50,n_estimators = 250)
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        #return model.score(y_test, y_pred)
        #return metrics.mean_absolute_error(y_test, y_pred)
        return metrics.r2_score(y_test, y_pred)
        #return np.sqrt(mean_squared_error(y_test, y_pred))


def csv_load(file):
        #file.save('data.csv')
        data = pd.read_csv('data.csv')
        #original_data = data.drop(data.columns[0],axis = 1)
        data.to_csv('data.csv',index=False)
        return data

def receive_data():
        data = pd.read_csv('data.csv')
        columns_name = []
        columns_nulldata =[]
        for colum in data.columns.values:
                columns_name.append(colum)
                if data[colum].isnull().sum() > 0:
                        columns_nulldata.append(colum)
        return data,columns_name,columns_nulldata

def serch_null(data):
        columns_nulldata = defaultdict(list)
        null_count = []
        columns_name = []
        columns_type = []
        for colum in data.columns.values:
                columns_name.append(colum)
        for colum in data.columns.values:
                if data[colum].isnull().sum() > 0:
                        columns_nulldata[colum].append(data[colum].isnull().sum())
                        columns_nulldata[colum].append(data[colum].dtype)
                #null_count.append(data[colum].isnull().sum())
                #if data[colum].isnull().sum() > 0:
                        #削除するかの処理
        #return data.dropna()
        return columns_nulldata,columns_name
        

def conv_object(data):
        conv_data,original_data = pd.factorize(data, sort=True)
        return conv_data


def conv_float(data):
        conv_data = data.astype('int')
        return conv_data

def conv_data(data):

        columns_type = []
        columns_name = data.columns.values
        #return columns_name,columns_type
        for colum in columns_name:
                if(data[colum].dtype == 'object'):#object
                        data[colum] = conv_object(data[colum])
                elif(data[colum].dtype == 'float'):#float
                        data[colum] = conv_float(data[colum])
                #elif(data[colum].dtype == 'int'):#int
                        #処理なし
                #else:#その他
        for colum in columns_name:
                columns_type.append(data[colum].dtype)
        return columns_name,columns_type

def ave(data):
  return data.fillna(data.mean())

def med(data):
  return data.fillna(data.median())

def mode(data):
  return data.fillna(max(data.mode()))

def standard(data):
  data_ave = data.mean()
  data_std = data.std()
  return data.fillna(np.random.randint(data_ave - data_std, data_ave + data_std))


def ml():
    batch_size =64
    n_classes = 10

    (X_train,y_train),(X_test,y_test) = mnist.load_data()

    X_train = X_train.reshape(60000,784)
    X_test = X_test.reshape(10000,784)
    X_train = X_train.astype(np.float32)/255
    X_test = X_test.astype(np.float32)/255
    y_train = keras.utils.to_categorical(y_train, n_classes)

    y_test = keras.utils.to_categorical(y_test, n_classes)

    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(784,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size,
            epochs=5,
            verbose=1,
            validation_split=0.1)

    score = model.evaluate(X_test, y_test, batch_size=batch_size)

    #print('Test accuracy:', score[1])
    return list(score)

def factorize(data):
        if data.dtype == 'object':
                data, uniques = pd.factorize(data)
        return data

#def ave(data):


def titanic(radio_data,target,model):
        #ave mode mean standard drop
        data = pd.read_csv("data.csv")
        columns_list = receive_data()[1]
        null_columns = receive_data()[2]
        for colum in columns_list:
                for null_column in null_columns:
                        if colum == null_column :
                                if radio_data[null_column] == "ave" :
                                        data[null_column] = ave(data[null_column])
                                elif radio_data[null_column] == "mode":
                                        data[null_column] = mode(data[null_column])
                                elif radio_data[null_column] == "med":
                                        data[null_column] = med(data[null_column])
                                elif radio_data[null_column] == "standard":
                                        data[null_column] = standard(data[null_column])
                                elif radio_data[null_column] == "drop":
                                        data.drop(null_column,axis=1,inplace=True)
                                        columns_list.remove(null_column)
        
        for colum in columns_list:
                data[colum] = factorize(data[colum])
                if radio_data[colum] == "drop":
                        data.drop(colum,axis=1,inplace=True)
        
        data.to_csv('data.csv',index=False)

        target_data = data[target]
        train_data = data.drop(target, axis = 1)
        x_train,x_test,y_train,y_test = model_selection.train_test_split(train_data,target_data,test_size = 0.2)

        #model = RandomForestClassifier()
        #model.fit(x_train,y_train)
        return choose_model(x_train,x_test,y_train,y_test,model)

def titanic_original(data):
        #ave mode mean standard drop
        data = pd.read_csv("data.csv")
        #null_columns = receive_data()[2]
        #for colum in receive_data()[1]:
        #        for null_column in null_columns:
        #                if data[colum] == data[null_column] :
        #                        if null_data[null_column] == "ave" :
        #                                data[null_column] = ave(data[null_column])
        #                        elif null_data[null_column] == "mode":
        #                                data[null_column] = mode(data[null_column])
        #                        elif null_data[null_column] == "med":
        #                                data[null_column] = mean(data[null_column])
        #                        elif null_data[null_column] == "standard":
        #                                data[null_column] = standard(data[null_column])
        #                        elif null_data[null_column] == "drop":
        #                                data.drop(null_column)

                                
        data['Sex'].replace(['male','female'], [0, 1], inplace=True)

        data['Embarked'].fillna(('S'), inplace=True)
        data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

        data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

        age_avg = data['Age'].mean()
        age_std = data['Age'].std()
        data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)

        delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
        data.drop(delete_columns, axis=1, inplace=True)

        #train = data[:len(train)]
        #test = data[len(train):]

        data = data.astype('int')

        target_data = data['Survived']
        train_data = data.drop('Survived', axis = 1)
        x_train,x_test,y_train,y_test = model_selection.train_test_split(train_data,target_data,test_size = 0.2)

        model = RandomForestClassifier(n_estimators=20,max_depth=35,criterion='entropy',warm_start=True)
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        return accuracy_score(y_test,y_pred)
