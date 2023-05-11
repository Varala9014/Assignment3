# -- coding: utf-8 --
"""
Created on Wed May 3 19:20:11 2023

@author: yashwanth reddy
"""

# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#reading the csv file using pandas
dataset = pd.read_csv("assig3.csv")


cluster_indi_codes = ["IQ.CPA.PUBS.XQ", "AG.LND.ARBL.ZS"]
cluster_years = ["2010", "2020"]

""" 
This function takes two arguments, `cluster_indi_codes` and 
`cluster_years`, and extracts the data from the dataset for a given Indicator
Code and given year. It then drops some columns from the dataset to create a 
new dataframe and returns a list of dataframes. 
"""
def Data_Cleaning(cluster_indi_codes, cluster_years):
    
    public_sector = dataset[dataset["Indicator Code"] == cluster_indi_codes[0]]
    arable_land = dataset[dataset["Indicator Code"] == cluster_indi_codes[1]]

    public_sector = public_sector.drop(
        ["Country Code", "Indicator Name", "Indicator Code"], axis=1).set_index("Country Name")
    arable_land = arable_land.drop(
        ["Country Code", "Indicator Name", "Indicator Code"], axis=1).set_index("Country Name")
    dfs = []
    for year in cluster_years:
        public_ = public_sector[year]
        arable_ = arable_land[year]
        data_frame = pd.DataFrame(
            {"CPIA public sector management ": public_, "Arable land": arable_})
        df = data_frame.dropna(axis=0)
        dfs.append(df)

    return dfs
