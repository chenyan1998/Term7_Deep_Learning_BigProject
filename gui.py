import streamlit as st
import pandas as pd
from datetime import datetime as dti
import numpy as np
import copy

import matplotlib.pyplot as plt
from ui_backend import *

st.set_option('deprecation.showPyplotGlobalUse', False)

def uiBackend():
    backend = Backend()
    backend.get_model_predictions()
    # st.write("Cache miss: ui backend")
    return backend

becopy = copy.deepcopy(uiBackend())

def getGraphs(dataset,plot_type):
    # plot cci
    fig, ax = plt.subplots()
    if plot_type=="cci":
        ax.plot(dataset.get_CCI(dataset.train, dataset.ndays1)[:100])
        ax.set_title("Commodity Channel Index (CCI)")
        ax.set_xlabel("Day")
        ax.set_ylabel("CCI")
    if plot_type=="macd":
        ax.plot(dataset.macd)
        ax.set_title("MACD Timeseries")
        ax.set_xlabel("Day")
        ax.set_ylabel("MACD")
    if plot_type=="signal":
        ax.plot(dataset.signal)
        ax.set_title("MACD signal Timeseries")
        ax.set_xlabel("Day")
        ax.set_ylabel("MACD signal")
    if plot_type=="hist":
        ax.plot(dataset.macd_hist)
        ax.set_title("MACD histogram Timeseries")
        ax.set_xlabel("Day")
        ax.set_ylabel("MACD histogram")
    return fig

def heatmap(data,sdate):
    cols = list(data.keys())
    info = list(data.values())
    y = list(range(1,len(info[0])+1))

    fig, ax = plt.subplots()
    im = ax.imshow(info)#harvest

    # show all ticks...
    ax.set_xticks(np.arange(len(y)))
    ax.set_yticks(np.arange(len(cols)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(y)
    ax.set_yticklabels(cols)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(cols)):
        for j in y:
            text = ax.text(j-1, i, round(info[i][j-1],2),
                    ha="center", va="center", color="w", size='xx-small')

    ax.set_title("stock data for "+sdate)
    fig.tight_layout()
    return fig

def main():
    st.title("50.039 Big Project Group 16")
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("",['Data Visualisation',"Test Set"])
    if choice != "Test Set":
        # display dataset
        st.subheader("Raw Training Dataset")
        train_data = pd.read_csv("./Data/SPY_train_2008_2020.csv")
        train_data
        
        # display graphs
        train1=train("./Data/SPY_train_2008_2016.csv")
        col1,col2 = st.beta_columns(2)
        col1.write(getGraphs(train1,"cci"))
        col2.write(getGraphs(train1,"macd"))
        col1.write(getGraphs(train1,"signal"))
        col2.write(getGraphs(train1,"hist"))

    else:
        st.subheader("Test Dataset (Load model and try here)")
        # load_model=torch.load("./models/train1model.pt")
        # test1=test("./Data/SPY_test_2016_2017.csv")
        test_data = pd.read_csv("./Data/SPY_test_2016_2017.csv")
        test_data
        # print(validation(load_model,test_loader,nn.CrossEntropyLoss()))
        # slider for display of each data
        selected_date = st.slider("Which date would you like to choose?",value=dti(2016,12,1),format="DD/MM/YYYY",min_value=dti(2016,1,4),max_value=dti(2017,12,29))
        sDate = str(selected_date).split(" ")[0]
        st.subheader("Selected date:" + sDate)
        newdf = test_data[test_data.Date==sDate]
        newdf
        if st.checkbox("Try with this data"):
            st.write("Results from",sDate)
            a = becopy.get_individual_data_sample(sDate)
            data={'High':a[0][0], 'Low':a[0][1], 'Close':a[0][2],\
                'CCI-5':a[0][3], 'CCI-10':a[0][4], 'CCI-20':a[0][5], \
                'MACD':a[0][6], 'MACD-Signal':a[0][7], 'MACD-History':a[0][8]}
            df = pd.DataFrame(data)
            st.subheader("Data Features")
            df.T
            # model prediction (int)
            st.header("Ground Truth Label: "+str(a[2]))
            st.header("Predicted Value: "+str(a[1]))
            if a[1] == 1:
                st.header("From the **model**,It is predicted to **RISE** the next day:sunglasses:")
            else:
                st.header("From the **model** , It is predicted to **FALL** the next day")
            fig=heatmap(data,sDate)
            st.write(fig)
        
        st.header("Trading Strategy")
        st.pyplot(becopy.get_strategy_plot())

main()
