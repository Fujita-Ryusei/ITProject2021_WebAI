
from io import TextIOWrapper
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
import xgboost as xgb
import seaborn as sns

def choose_model(x_train,x_test,y_train,y_test,model,param):
        if model == "RandomForestClassifier":
                return RFC(x_train,x_test,y_train,y_test,param)
        elif model == "RandomForestRegressor":
                return RFR(x_train,x_test,y_train,y_test,param)
        elif model == "XGBoost":
                return XGB(x_train,x_test,y_train,y_test,param)

def param_none(param):
        if(type(param) is int):
                return int(param)
        else:
                return None

def RFC(x_train,x_test,y_train,y_test,param):
        model = RandomForestClassifier(
                n_estimators=int(param[0]),criterion=param[1],max_depth=param_none(param[2]),
                min_samples_split=int(param[3]),max_leaf_nodes=param_none(param[4])
                )
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        return accuracy_score(y_test,y_pred)

def RFR(x_train,x_test,y_train,y_test,param):
        #model = LogisticRegression()
        model = RandomForestRegressor(
                n_estimators=int(param[0]),criterion=param[1],max_depth=param_none(param[2]),
                min_samples_split=int(param[3]),max_leaf_nodes=param_none(param[4])
                )
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        #return model.score(x_test,y_test)
        #return metrics.mean_absolute_error(y_test, y_pred)
        return metrics.r2_score(y_test, y_pred)
        #return np.sqrt(mean_squared_error(y_test, y_pred))

def XGB(x_train,x_test,y_train,y_test,radio_param):
        train = xgb.DMatrix(x_train, label=y_train)
        if(radio_param[2] == "reg:linear"):
                param = {'max_depth': int(radio_param[0]), 'eta': float(radio_param[1]), 'objective': radio_param[2] }
                num_round = int(radio_param[3])
                bst = xgb.train(param, train, num_round)
                test = xgb.DMatrix(x_test)
                y_pred = bst.predict(test)
                return metrics.r2_score(y_test, y_pred)
        elif(radio_param[2] == "multi:softmax"):
                param = {'max_depth': int(radio_param[0]), 'eta': float(radio_param[1]), 'objective': radio_param[2] , 'num_class': 3}
                num_round = int(radio_param[3])
                bst = xgb.train(param, train, num_round)
                test = xgb.DMatrix(x_test)
                y_pred = bst.predict(test)
                return accuracy_score(y_test, y_pred)
        else:
                raise Exception("XGBoostのobjectiveの選択に間違いがあります")
        #reg = xgb.XGBRegressor()
        ##学習過程を表示するための変数を用意
        #reg.fit(x_train,y_train,eval_set=[(x_train,y_train),(x_test,y_test)])
        #y_pred = reg.predict(x_test)
        #return metrics.r2_score(y_test, y_pred)

def csv_load(file):
        #file.save('data.csv')
        data = pd.read_csv('data.csv')
        #original_data = data.drop(data.columns[0],axis = 1)
        #data.to_csv('data.csv',index=False)
        return data

def img_test(data):
        line_plot = sns.barplot(x='Survived', y='Age', data=data)
        figure = line_plot.get_figure()
        figure.savefig("static/img/gr.jpg")


######receiveとserchをひとつにする

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
        try:
                conv_data,original_data = pd.factorize(data, sort=True)
                return conv_data
        except Exception as e:
                raise Exception("Objectを変換できませんでした")


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



def factorize(data):
        if data.dtype == 'object':
                data, uniques = pd.factorize(data)
        return data



def ml(radio_data,target,model,param):
        #ave mode mean standard drop
        data = pd.read_csv("data.csv")
        #img_test(data)
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
                if radio_data[colum] == "drop" and colum != target:
                        data.drop(colum,axis=1,inplace=True)
        
        #data.to_csv('data.csv',index=False)

        target_data = data[target]
        train_data = data.drop(target, axis = 1)

        

        x_train,x_test,y_train,y_test = model_selection.train_test_split(train_data,target_data,test_size = 0.2)

        #model = RandomForestClassifier()
        #model.fit(x_train,y_train)
        return choose_model(x_train,x_test,y_train,y_test,model,param)


