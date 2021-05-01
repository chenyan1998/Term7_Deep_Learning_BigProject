import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class TradingStrategy:
    def __init__(self, model, test_loader, df):
        #get model
        self.model = model
        
        #get dataloader for timeframe to test on
        self.test_loader = test_loader
    
        #get original trading period dataframe
        self.df = df
        
        #run function to get model predictions
        self.get_model_predictions()
        
        #run function to generate trade df
        self.generate_trade_period_df()
        
    def get_model_predictions(self):
        y_true = []
        y_pred = []
        model_confidence = []
        with torch.no_grad():
            for i, (inputs, classes) in enumerate(self.test_loader):
                outputs = self.model(inputs)
                scores, preds = torch.max(outputs, 1)

                model_confidence += scores.tolist()
                y_pred += (preds.tolist())
                y_true += [i[0] for i in classes.tolist()]

        self.y_pred = y_pred
        self.y_true = y_true
        self.model_confidence = model_confidence
    
    def generate_trade_period_df(self):
        self.df["prediction"] = self.y_pred
        self.df["truth"] = self.y_true
        self.df["model_confidence"] = self.model_confidence
        self.df["model_confidence_average"] = self.df["model_confidence"].expanding().mean()
        
    def strategy_basic(self):
        trade_period1 = self.df
        
        #function will return: profits, previous_prices
        units_bought = 0

        price_of_unitbought = 0

        profits = []
        previous_prices = []

        #naive strategy that buys if prediction is 1, sells if prediction is 0,
        #but only if no other trades are open
        for i in range(len(trade_period1)):
            if i == len(trade_period1):
                if units_bought == 1:
                    #close previous long
                    profits.append(trade_period1["Close"][i] - price_of_unitbought)
                    previous_prices.append(price_of_unitbought)
                    units_bought = 0
                elif units_bought == -1:
                    #close previous short
                    profits.append(price_of_unitbought - trade_period1["Close"][i])
                    previous_prices.append(price_of_unitbought)
                    units_bought = 0
                break
            #we only open new trades if we dont have any previous trades
            if units_bought == 0:
                if((trade_period1["prediction"][i] == 1)):
                    #buy
                    price_of_unitbought = trade_period1["Close"][i]
                    profits.append(0)
                    previous_prices.append(0)
                    units_bought = 1
                elif((trade_period1["prediction"][i] == 0)):
                    #sell
                    price_of_unitbought = trade_period1["Close"][i]
                    profits.append(0)
                    previous_prices.append(0)
                    units_bought = -1
            elif((units_bought == 1) and (trade_period1["prediction"][i] == 0)):
                #close previous long
                profits.append(trade_period1["Close"][i] - price_of_unitbought)
                previous_prices.append(price_of_unitbought)
                units_bought = 0

            elif((units_bought == -1) and (trade_period1["prediction"][i] == 1)):
                #close previous short
                profits.append(price_of_unitbought - trade_period1["Close"][i])
                previous_prices.append(price_of_unitbought)
                units_bought = 0
            else:
                profits.append(0)
                previous_prices.append(0)

        return((profits, previous_prices))
    
    def strategy_unlimited_buying_selling(self, stock_threshold = 2):
        trade_period1 = self.df
        
        #function will return: profits, previous_prices, dates
        
        dates = []

        price_of_unitbought = []
        price_of_unitsold = []

        profits = []
        previous_prices = []

        """
        columns to use:
        1. Close: price
        2. Prediction: model pred        
        """
        
        #expanded version of previous strategy that buys and sells whenever signal is favourable
        for i in range(len(trade_period1)):
            #case where we're at end of trading period: close all open trades
            if i == len(trade_period1)-1:
                for j in range(len(price_of_unitsold)):
                    #close previous short positions
                    profits.append(price_of_unitsold[j] - trade_period1["Close"][i])
                    previous_prices.append(price_of_unitsold[j])
                    dates.append(trade_period1["Date"][i])
                for j in range(len(price_of_unitbought)):
                    #close previous long positions
                    profits.append(trade_period1["Close"][i] - price_of_unitbought[j])
                    previous_prices.append(price_of_unitbought[j])
                    dates.append(trade_period1["Date"][i])
                break
            
            if trade_period1["prediction"][i] == 1:

                
                #check if there are units sold
                if len(price_of_unitsold)>0:
                    #close previous trade
                    profits.append(price_of_unitsold[0] - trade_period1["Close"][i])
                    previous_prices.append(price_of_unitsold[0])
                    dates.append(trade_period1["Date"][i])
                    #update price_of_unitsold array
                    price_of_unitsold = price_of_unitsold[1:]
                
                elif len(price_of_unitbought)>stock_threshold:
                    #not buying anything
                    profits.append(0)
                    previous_prices.append(0)
                    dates.append(trade_period1["Date"][i])
                
                #elif np.random.uniform() < 0.3:
                else:
                    #buy
                    #open new long trade
                    price_of_unitbought.append(trade_period1["Close"][i])
                    profits.append(0)
                    previous_prices.append(0)
                    dates.append(trade_period1["Date"][i])
            
            elif trade_period1["prediction"][i] == 0:
                #sell
                #check if there are units bought before
                if len(price_of_unitbought)>0:
                    #close previous trade
                    profits.append(trade_period1["Close"][i] - price_of_unitbought[0])
                    previous_prices.append(price_of_unitbought[0])
                    dates.append(trade_period1["Date"][i])
                    #update price_of_unitbought array
                    price_of_unitbought = price_of_unitbought[1:]
                
                elif len(price_of_unitsold)>stock_threshold:
                    #not selling anything
                    profits.append(0)
                    previous_prices.append(0)
                    dates.append(trade_period1["Date"][i])
                
                else:
                    #open new short trade
                    price_of_unitsold.append(trade_period1["Close"][i])
                    profits.append(0)
                    previous_prices.append(0)
                    dates.append(trade_period1["Date"][i])

        return((profits, previous_prices, dates))

    def strategy_moderate_confidence_based(self, stock_threshold = 2):
        trade_period1 = self.df
        
        #function will return: profits, previous_prices, dates
        
        dates = []

        price_of_unitbought = []
        price_of_unitsold = []

        profits = []
        previous_prices = []

        """
        columns to use:
        1. Close: price
        2. Prediction: model pred        
        """
        
        #expanded version of previous strategy that buys and sells whenever signal is favourable
        for i in range(len(trade_period1)):
            #case where we're at end of trading period: close all open trades
            if i == len(trade_period1)-1:
                for j in range(len(price_of_unitsold)):
                    #close previous short positions
                    profits.append(price_of_unitsold[j] - trade_period1["Close"][i])
                    previous_prices.append(price_of_unitsold[j])
                    dates.append(trade_period1["Date"][i])
                for j in range(len(price_of_unitbought)):
                    #close previous long positions
                    profits.append(trade_period1["Close"][i] - price_of_unitbought[j])
                    previous_prices.append(price_of_unitbought[j])
                    dates.append(trade_period1["Date"][i])
                break
            
            if(trade_period1["prediction"][i] == 1 and (trade_period1["model_confidence"][i] > trade_period1["model_confidence_average"][i])):
                #throw a dice, buy only 42% of the time, as the model predicts way too many 'buys'
                #details: 30% of the model predictions are 0, 70% are 1. To normalize this, we only go act on the buy signal
                # if there are short sales to be closed, or throw the dice
                
                #check if there are units sold
                if len(price_of_unitsold)>0:
                    #close previous trade
                    profits.append(price_of_unitsold[0] - trade_period1["Close"][i])
                    previous_prices.append(price_of_unitsold[0])
                    dates.append(trade_period1["Date"][i])
                    #update price_of_unitsold array
                    price_of_unitsold = price_of_unitsold[1:]
                
                elif len(price_of_unitbought)>stock_threshold:
                    #not buying anything
                    profits.append(0)
                    previous_prices.append(0)
                    dates.append(trade_period1["Date"][i])
                
                #elif np.random.uniform() < 0.3:
                else:
                    #buy
                    #open new long trade
                    price_of_unitbought.append(trade_period1["Close"][i])
                    profits.append(0)
                    previous_prices.append(0)
                    dates.append(trade_period1["Date"][i])
            
            #(trade_period1["model_confidence"][i] > trade_period1["model_confidence_average"][i])
            elif(trade_period1["prediction"][i] == 0):
                #sell
                #check if there are units bought before
                if len(price_of_unitbought)>0:
                    #close previous trade
                    profits.append(trade_period1["Close"][i] - price_of_unitbought[0])
                    previous_prices.append(price_of_unitbought[0])
                    dates.append(trade_period1["Date"][i])
                    #update price_of_unitbought array
                    price_of_unitbought = price_of_unitbought[1:]
                
                elif len(price_of_unitsold)>stock_threshold:
                    #not selling anything
                    profits.append(0)
                    previous_prices.append(0)
                    dates.append(trade_period1["Date"][i])
                
                else:
                    #open new short trade
                    price_of_unitsold.append(trade_period1["Close"][i])
                    profits.append(0)
                    previous_prices.append(0)
                    dates.append(trade_period1["Date"][i])
            else:
                #not selling or buying anything
                profits.append(0)
                previous_prices.append(0)
                dates.append(trade_period1["Date"][i])

        return((profits, previous_prices, dates))
    
class VisualizeResults:
    def __init__(self, profits, previous_prices, closing_prices):
        self.profits = profits
        self.previous_prices = previous_prices
        self.closing_prices = closing_prices
        self.calculate_returns()
    
    def calculate_returns(self):
        strategy_returns = []
        for i in range(len(self.profits)):
            if self.previous_prices[i]==0:
                strategy_returns.append(0)
                continue
            else:
                strategy_returns.append(self.profits[i]/self.previous_prices[i])
        self.strategy_returns = strategy_returns
    
    def visualize_individual_returns(self, trade_period_dates=None):
        if trade_period_dates is None:
            trade_period_dates = range(len(self.strategy_returns))
        plt.figure(figsize=(20,10))
        plt.plot(trade_period_dates, self.strategy_returns)
        plt.title("Individual Trade Returns")
        plt.xlabel("Date")
        plt.xticks(rotation=70)
        ax = plt.gca()
        counter = 0
        for label in ax.get_xaxis().get_ticklabels()[::]:
            if counter % 9 == 0:
                pass
            else:
                label.set_visible(False)
            counter += 1
    
    def visualize_cumulative_returns(self, trade_period_dates=None):
        if trade_period_dates is None:
            trade_period_dates = range(len(self.strategy_returns))
        plt.figure(figsize=(20,10))
        plt.plot(trade_period_dates, np.cumprod(np.array(self.strategy_returns) + 1))
        plt.xticks(rotation=70)
        ax = plt.gca()
        counter = 0
        for label in ax.get_xaxis().get_ticklabels()[::]:
            if counter % 9 == 0:
                pass
            else:
                label.set_visible(False)
            counter += 1
    
    def visualize_trading_benchmark_performance(self, trade_period_dates=None):
        if trade_period_dates is None:
            trade_period_dates = range(len(self.strategy_returns))
        plt.figure(figsize=(20,10))
        plt.plot(trade_period_dates, self.closing_prices[0]*np.cumprod(1 + np.array(self.strategy_returns)), label="Trading Strategy")
        plt.plot(trade_period_dates, self.closing_prices, label="Buy and Hold")
        plt.legend()
        plt.xticks(rotation=70)
        ax = plt.gca()
        counter = 0
        for label in ax.get_xaxis().get_ticklabels()[::]:
            if counter % 9 == 0:
                pass
            else:
                label.set_visible(False)
            counter += 1