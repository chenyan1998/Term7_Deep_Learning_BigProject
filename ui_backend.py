import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
import datetime

from dataset import train, val, test
from helper_functions import plot_loss, plot_accuracy, validation, cal_acc

import trading_strategy as ts

class StockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.padding=nn.ZeroPad2d(1)
        self.conv1=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3),
            nn.ReLU()
        )
        
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,kernel_size=(3, 2)),
            nn.ReLU()
        )
        
        self.linear=nn.Linear(32*8*2,2)
        
        self.act=nn.Softmax()
        
        self.history={'train_accuracy':[],'train_loss':[],'validation_accuracy':[],'validation_loss':[]}
        
        
    def forward(self,x):
#         print(x.size())
        out=self.padding(x)
        out=self.conv1(out)
        out=self.conv2(out)
#         print(out.size())
        out=out.view(out.size(0),-1)
        out=self.linear(out)
        out=self.act(out) 
#         print(out.size())
        return out

def get_model_predictions(model, test_loader):
    y_true = []
    y_pred = []

    nb_classes = 2

    confusion_matrix_ = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(test_loader):
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            y_pred += (preds.tolist())
            y_true += [i[0] for i in classes.tolist()]

    return((y_pred, y_true))

#here test1 is the test class
#inputs: y_pred, y_true, test object
#outputs: model_predictions df, test_dates
def combine_model_predictions(y_pred, y_true, test1):
    test1_dates = test1.train_date
    test1_price_df = test1.data

    model_predictions = test1_price_df.loc[test1_price_df['Date'].isin(test1_dates)].reset_index(drop=True)
    model_predictions["y_pred"] = y_pred
    model_predictions["y_true"] = y_true
    
    return model_predictions, test1_dates

def get_image_inputdata(image):
    high = []
    low = []
    close = []

    cci5 = []
    cci10 = []
    cci20 = []

    macd = []
    macdsignal = []
    macdhist = []

    for index in range(3):
        two_d = image[index]
        for j in range(len(two_d)):
            row = two_d[j]
            if index == 0:
                high.append(row[0])
                cci5.append(row[1])
                macd.append(row[2])
            elif index == 1:
                low.append(row[0])
                cci10.append(row[1])
                macdsignal.append(row[2])
            else:
                close.append(row[0])
                cci20.append(row[1])
                macdhist.append(row[2])
                
    return((high, low, close, cci5, cci10, cci20, macd, macdsignal, macdhist))

def get_input_output(test1, model_predictions, test1_dates, date):
    date_index = test1_dates.reset_index(drop=True).index[test1_dates == date][0]
    image = test1.__getitem__(date_index)[0].tolist()
    
    high, low, close, cci5, cci10, cci20, macd, macdsignal, macdhist = get_image_inputdata(image)
    input_data = ((high, low, close, cci5, cci10, cci20, macd, macdsignal, macdhist))
    
    model_pred = model_predictions["y_pred"][date_index]
    y_true = model_predictions["y_true"][date_index]
    
    return input_data, model_pred, y_true

class Backend:
    def __init__(self):
        self.bs_val=10
        self.train1=train("./Data/SPY_train_2008_2016.csv")
        self.test1=test("./Data/SPY_test_2016_2017.csv")
        self.train1_loader=DataLoader(self.train1, batch_size=self.bs_val, shuffle=False)
        self.test1_loader=DataLoader(self.test1, batch_size=self.bs_val, shuffle=False)
        
        self.train2=train("./Data/SPY_train_2008_2020.csv")
        self.test2=test("./Data/SPY_test_2020_2021.csv")
        self.train2_loader=DataLoader(self.train2, batch_size=self.bs_val, shuffle=False)
        self.test2_loader=DataLoader(self.test2, batch_size=self.bs_val, shuffle=False)
        
        self.model=torch.load("./models/train_final.pt")
    
    def get_model_predictions(self):
        self.y_pred1, self.y_true1 = get_model_predictions(self.model, self.test1_loader)
        self.model_predictions1, self.test1_dates = combine_model_predictions(self.y_pred1, self.y_true1, self.test1)
        
        self.y_pred2, self.y_true2 = get_model_predictions(self.model, self.test2_loader)
        self.model_predictions2, self.test2_dates = combine_model_predictions(self.y_pred2, self.y_true2, self.test2)
    
    def get_individual_data_sample(self, date):
        input_data1, model_pred1, y_true1 = get_input_output(self.test1, self.model_predictions1, self.test1_dates, date)
        return((input_data1, model_pred1, y_true1))
    
    def get_strategy_plot(self):
        trade = ts.TradingStrategy(self.model, self.test1_loader, self.model_predictions1)
        profits, prev_prices, dates = trade.strategy_unlimited_buying_selling(stock_threshold=1)
        closing_prices = self.model_predictions1["Close"]

        #pass profits and previous prices to visualizer
        visualizer = ts.VisualizeResults(profits[1:], prev_prices[1:], closing_prices.reset_index(drop=True))
        visualizer.visualize_trading_benchmark_performance(self.model_predictions1["Date"])