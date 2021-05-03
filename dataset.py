import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class train(Dataset):
    '''
    This is the class for training dataset
    The class will load the training dataset, process the original data to get image data for training model 
    '''
    def __init__(self,path):
        self.dataset_paths=path
        
        self.price_features = ["High", "Low", "Close"]
        self.CCI_features = ["CCI5", "CCI10", "CCI20"]
        self.MACD_features = ["MACD", "MACDsignal", "MACDhist"]
        
        self.data=pd.read_csv(self.dataset_paths)
        self.train=self.data[:int(0.95*len(self.data))]
    
        self.ndays1=5
        self.ndays2=10
        self.ndays3=20
        
        self.macd, self.signal, self.macd_hist = self.get_macd_indicators(self.train)
        self.train_date, self.train_images, self.train_labels=self.get_features()
        
    def get_CCI(self,data, ndays):
        '''
        The function calculates the CCI values of our dataset
        '''
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3 
        moving_avg = typical_price.rolling(ndays).mean()
        mean_deviation = typical_price.rolling(ndays).apply(lambda x: pd.Series(x).mad())
        CCI = (typical_price - moving_avg) / (0.015 * mean_deviation) 
        return CCI
    
    def get_macd_indicators(self,data):
        '''
        The function calculates the MACD values of our dataset
        '''
        moving_avg26 = data['Close'].rolling(26).mean()
        exp_moving_avg12 = data['Close'].ewm(span=12, adjust=False).mean()
        macd = exp_moving_avg12 - moving_avg26
        signal = macd.rolling(9).mean()
        macd_hist = macd - signal
        return macd, signal, macd_hist
    
    def get_train_images(self,train_features):
        '''
        The function produce the image data that we need for our training
        '''
        train_images = []
        for i in range(len(train_features)-9):
            image1 = train_features[self.price_features][i:i+10].T
            image2 = train_features[self.CCI_features][i:i+10].T
            image3 = train_features[self.MACD_features][i:i+10].T
            #image4 = train_features[return_features][i:i+10].T


            image_total1 = np.append(arr=image1.values.reshape((3, 10, 1)), values=image2.values.reshape((3, 10, 1)), axis=2)
            image_total2 = np.append(arr=image_total1, values=image3.values.reshape((3, 10, 1)), axis=2)
            #image_total3 = np.append(arr=image_total2, values=image4.values.reshape((3, 10, 1)), axis=2)

            train_images.append(torch.Tensor(image_total2))

        return train_images
    
    def plot_CCI(self):
        '''
        The function plot CCI values of the dataset
        '''
        fig, ax = plt.subplots()
        ax.plot(self.get_CCI(self.train, self.ndays1)[:100])
        plt.show()
        return None
    
    def plot_macd(self):
        '''
        The function plot MACD values of the dataset
        '''
        fig,ax=plt.subplots()
        ax.plot(self.macd)
        ax.set_title("MACD Timeseries")
        ax.set_xlabel("Day")
        ax.set_ylabel("MACD")
        plt.show()
        return None
        
    def plot_signal(self):
        '''
        The function plot signal values of the dataset
        '''
        fig,ax=plt.subplots()
        ax.plot(self.signal)
        ax.set_title("MACD signal Timeseries")
        ax.set_xlabel("Day")
        ax.set_ylabel("MACD signal")
        plt.show()
        return None
        
    def plot_macd_hist(self):
        '''
        The function plot MACD histogram of the dataset
        '''
        fig,ax=plt.subplots()
        ax.plot(self.macd_hist)
        ax.set_title("MACD histogram Timeseries")
        ax.set_xlabel("Day")
        ax.set_ylabel("MACD histogram")
        plt.show()
        return None
        
    def __len__(self):
        return len(self.train_images)
    
    def get_features(self):
        train_features = self.train.copy()
        train_features["CCI5"] = self.get_CCI(train_features, 5)
        train_features["CCI10"] = self.get_CCI(train_features, 10)
        train_features["CCI20"] = self.get_CCI(train_features, 20)

        train_features["MACD"], train_features["MACDsignal"], train_features["MACDhist"] = self.get_macd_indicators(train_features)

        #Calculate future returns features
        close_next = train_features["Close"].shift(-1)
        returns = (close_next - train_features["Close"]) / (train_features["Close"])

        #convert futures return to boolean
        train_features["y"] = np.sign(returns).replace(-1, 0)

        #train_features["returns1"] = returns.shift(1)
        #train_features["returns2"] = returns.shift(2)
        #train_features["returns3"] = returns.shift(3)

        train_features = train_features.dropna()
        train_date = train_features["Date"]
        train_labels = train_features["y"]
        train_features = train_features[self.price_features + self.CCI_features + self.MACD_features]

        #convert train_features and train_labels to train_images in tf format
        train_images = torch.stack(tuple(self.get_train_images(train_features)),dim=0)
        train_labels = torch.stack(tuple([torch.Tensor([i]) for i in train_labels[9:]]),dim=0)
#         print(train_images.size())
#         print(train_images)
#         print(train_labels[9:])
#         print(train_labels)
#         train_images = self.get_train_images(train_features)
#         train_labels = train_labels[9:]
       


        return train_date[9:], train_images, train_labels
    
    def __getitem__(self,index):
#         print(index)
        
#         print(self.train_images[0])
        return self.train_images[index],self.train_labels[index]




class val(Dataset):
    '''
    This is the class for validation dataset
    The class will load the validation dataset, process the original data to get image data for training model 
    '''
    def __init__(self,path):
        self.dataset_paths=path
        
        self.price_features = ["High", "Low", "Close"]
        self.CCI_features = ["CCI5", "CCI10", "CCI20"]
        self.MACD_features = ["MACD", "MACDsignal", "MACDhist"]
        
        self.data=pd.read_csv(self.dataset_paths)
        self.train=self.data[int(0.95*len(self.data)):]
    
        self.ndays1=5
        self.ndays2=10
        self.ndays3=20
        
        self.macd, self.signal, self.macd_hist = self.get_macd_indicators(self.train)
        self.train_date, self.train_images, self.train_labels=self.get_features()
#         print(self.train_images.size())
        
    def get_CCI(self,data, ndays):
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3 
        moving_avg = typical_price.rolling(ndays).mean()
        mean_deviation = typical_price.rolling(ndays).apply(lambda x: pd.Series(x).mad())
        CCI = (typical_price - moving_avg) / (0.015 * mean_deviation) 
        return CCI
    
    def get_macd_indicators(self,data):
        moving_avg26 = data['Close'].rolling(26).mean()
        exp_moving_avg12 = data['Close'].ewm(span=12, adjust=False).mean()
        macd = exp_moving_avg12 - moving_avg26
        signal = macd.rolling(9).mean()
        macd_hist = macd - signal
        return macd, signal, macd_hist
    
    def get_train_images(self,train_features):
        train_images = []
        for i in range(len(train_features)-9):
            image1 = train_features[self.price_features][i:i+10].T
            image2 = train_features[self.CCI_features][i:i+10].T
            image3 = train_features[self.MACD_features][i:i+10].T
            #image4 = train_features[return_features][i:i+10].T


            image_total1 = np.append(arr=image1.values.reshape((3, 10, 1)), values=image2.values.reshape((3, 10, 1)), axis=2)
            image_total2 = np.append(arr=image_total1, values=image3.values.reshape((3, 10, 1)), axis=2)
            #image_total3 = np.append(arr=image_total2, values=image4.values.reshape((3, 10, 1)), axis=2)

            train_images.append(torch.Tensor(image_total2))

        return train_images
    
    def plot_CCI(self):
        fig, ax = plt.subplots()
        ax.plot(self.get_CCI(self.train, self.ndays1)[:100])
        plt.show()
        return None
    
    def plot_macd(self):
        fig,ax=plt.subplots()
        ax.plot(self.macd)
        ax.set_title("MACD Timeseries")
        ax.set_xlabel("Day")
        ax.set_ylabel("MACD")
        plt.show()
        return None
        
    def plot_signal(self):
        fig,ax=plt.subplots()
        ax.plot(self.signal)
        ax.set_title("MACD signal Timeseries")
        ax.set_xlabel("Day")
        ax.set_ylabel("MACD signal")
        plt.show()
        return None
        
    def plot_macd_hist(self):
        fig,ax=plt.subplots()
        ax.plot(self.macd_hist)
        ax.set_title("MACD histogram Timeseries")
        ax.set_xlabel("Day")
        ax.set_ylabel("MACD histogram")
        plt.show()
        return None
        
    def __len__(self):
        return len(self.train_images)
    
    def get_features(self):
        train_features = self.train.copy()
        train_features["CCI5"] = self.get_CCI(train_features, 5)
        train_features["CCI10"] = self.get_CCI(train_features, 10)
        train_features["CCI20"] = self.get_CCI(train_features, 20)

        train_features["MACD"], train_features["MACDsignal"], train_features["MACDhist"] = self.get_macd_indicators(train_features)

        #Calculate future returns features
        close_next = train_features["Close"].shift(-1)
        returns = (close_next - train_features["Close"]) / (train_features["Close"])

        #convert futures return to boolean
        train_features["y"] = np.sign(returns).replace(-1, 0)

        #train_features["returns1"] = returns.shift(1)
        #train_features["returns2"] = returns.shift(2)
        #train_features["returns3"] = returns.shift(3)

        train_features = train_features.dropna()
        train_date = train_features["Date"]
        train_labels = train_features["y"]
        train_features = train_features[self.price_features + self.CCI_features + self.MACD_features]

        #convert train_features and train_labels to train_images in tf format
        train_images = torch.stack(tuple(self.get_train_images(train_features)),dim=0)
        train_labels = torch.stack(tuple([torch.Tensor([i]) for i in train_labels[9:]]),dim=0)
#         print(train_images.size())
#         print(train_images)
#         print(train_labels[9:])
#         print(train_labels)
#         train_images = self.get_train_images(train_features)
#         train_labels = train_labels[9:]
       


        return train_date[9:], train_images, train_labels
    
    def __getitem__(self,index):
        
#         print(self.train_images[0])
        return self.train_images[index],self.train_labels[index]



class test(Dataset):
    '''
    This is the class for testing dataset
    The class will load the testing dataset, process the original data to get image data
    '''
    def __init__(self,path):
        self.dataset_paths=path
        
        self.price_features = ["High", "Low", "Close"]
        self.CCI_features = ["CCI5", "CCI10", "CCI20"]
        self.MACD_features = ["MACD", "MACDsignal", "MACDhist"]
        
        self.data=pd.read_csv(self.dataset_paths)
        self.train=self.data[:int(0.95*len(self.data))]
    
        self.ndays1=5
        self.ndays2=10
        self.ndays3=20
        
        self.macd, self.signal, self.macd_hist = self.get_macd_indicators(self.train)
        self.train_date, self.train_images, self.train_labels=self.get_features()
        
    def get_CCI(self,data, ndays):
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3 
        moving_avg = typical_price.rolling(ndays).mean()
        mean_deviation = typical_price.rolling(ndays).apply(lambda x: pd.Series(x).mad())
        CCI = (typical_price - moving_avg) / (0.015 * mean_deviation) 
        return CCI
    
    def get_macd_indicators(self,data):
        moving_avg26 = data['Close'].rolling(26).mean()
        exp_moving_avg12 = data['Close'].ewm(span=12, adjust=False).mean()
        macd = exp_moving_avg12 - moving_avg26
        signal = macd.rolling(9).mean()
        macd_hist = macd - signal
        return macd, signal, macd_hist
    
    def get_train_images(self,train_features):
        train_images = []
        for i in range(len(train_features)-9):
            image1 = train_features[self.price_features][i:i+10].T
            image2 = train_features[self.CCI_features][i:i+10].T
            image3 = train_features[self.MACD_features][i:i+10].T
            #image4 = train_features[return_features][i:i+10].T


            image_total1 = np.append(arr=image1.values.reshape((3, 10, 1)), values=image2.values.reshape((3, 10, 1)), axis=2)
            image_total2 = np.append(arr=image_total1, values=image3.values.reshape((3, 10, 1)), axis=2)
            #image_total3 = np.append(arr=image_total2, values=image4.values.reshape((3, 10, 1)), axis=2)

            train_images.append(torch.Tensor(image_total2))

        return train_images
    
    def plot_CCI(self):
        fig, ax = plt.subplots()
        ax.plot(self.get_CCI(self.train, self.ndays1)[:100])
        plt.show()
        return None
    
    def plot_macd(self):
        fig,ax=plt.subplots()
        ax.plot(self.macd)
        ax.set_title("MACD Timeseries")
        ax.set_xlabel("Day")
        ax.set_ylabel("MACD")
        plt.show()
        return None
        
    def plot_signal(self):
        fig,ax=plt.subplots()
        ax.plot(self.signal)
        ax.set_title("MACD signal Timeseries")
        ax.set_xlabel("Day")
        ax.set_ylabel("MACD signal")
        plt.show()
        return None
        
    def plot_macd_hist(self):
        fig,ax=plt.subplots()
        ax.plot(self.macd_hist)
        ax.set_title("MACD histogram Timeseries")
        ax.set_xlabel("Day")
        ax.set_ylabel("MACD histogram")
        plt.show()
        return None
        
    def __len__(self):
        return len(self.train_images)
    
    def get_features(self):
        train_features = self.train.copy()
        train_features["CCI5"] = self.get_CCI(train_features, 5)
        train_features["CCI10"] = self.get_CCI(train_features, 10)
        train_features["CCI20"] = self.get_CCI(train_features, 20)

        train_features["MACD"], train_features["MACDsignal"], train_features["MACDhist"] = self.get_macd_indicators(train_features)

        #Calculate future returns features
        close_next = train_features["Close"].shift(-1)
        returns = (close_next - train_features["Close"]) / (train_features["Close"])

        #convert futures return to boolean
        train_features["y"] = np.sign(returns).replace(-1, 0)

        #train_features["returns1"] = returns.shift(1)
        #train_features["returns2"] = returns.shift(2)
        #train_features["returns3"] = returns.shift(3)

        train_features = train_features.dropna()
        train_date = train_features["Date"]
        train_labels = train_features["y"]
        train_features = train_features[self.price_features + self.CCI_features + self.MACD_features]

        #convert train_features and train_labels to train_images in tf format
        train_images = torch.stack(tuple(self.get_train_images(train_features)),dim=0)
        train_labels = torch.stack(tuple([torch.Tensor([i]) for i in train_labels[9:]]),dim=0)
#         print(train_images.size())
#         print(train_images)
#         print(train_labels[9:])
#         print(train_labels)
#         train_images = self.get_train_images(train_features)
#         train_labels = train_labels[9:]
       


        return train_date[9:], train_images, train_labels
    
    def __getitem__(self,index):
#         print(index)
        
#         print(self.train_images[0])
        return self.train_images[index],self.train_labels[index]
        
        