import pandas as pd
import numpy as np
import featuretools as ft
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from featuretools.primitives import Count, Mean
import os

def get_country_df(): 

    def p2f(x):
        """
        Convert urban percentage to float
        """
        try:
            return float(x.strip('%'))/100
        except:
            return np.nan

    def age2int(x):
        """
        Convert Age to integer
        """
        try:
            return int(x)
        except:
            return np.nan

    def fert2float(x):
        """
        Convert Fertility Rate to float
        """
        try:
            return float(x)
        except:
            return np.nan


    countries_df = pd.read_csv("./data/population_by_country_2020.csv", converters={'Urban Pop %':p2f,
                                                                                    'Fert. Rate':fert2float,
                                                                                    'Med. Age':age2int})
    countries_df.rename(columns={'Country (or dependency)': 'country',
                                 'Population (2020)' : 'population',
                                 'Density (P/Km²)' : 'density',
                                 'Fert. Rate' : 'fertility',
                                 'Med. Age' : "age",
                                 'Urban Pop %' : 'urban_percentage'}, inplace=True)



    countries_df['country'] = countries_df['country'].replace('United States', 'US')
    countries_df = countries_df[["country", "population", "density", "fertility", "age", "urban_percentage"]]
    return countries_df

def get_weather_df():
    df_temperature = pd.read_csv('data/temperature_dataframe.csv')
    df_temperature['country'] = df_temperature['country'].replace('USA', 'US')
    df_temperature['country'] = df_temperature['country'].replace('UK', 'United Kingdom')
    df_temperature = df_temperature[["country", "province", "date", "humidity", "sunHour", "tempC", "windspeedKmph"]].reset_index()
    df_temperature.rename(columns={'province': 'state'}, inplace=True)
    df_temperature["date"] = pd.to_datetime(df_temperature['date'])
    df_temperature['state'] = df_temperature['state'].fillna('')

    return df_temperature

def get_icu_df():
    icu_df = pd.read_csv("./data/icu_bed.csv")
    icu_df['Country Name'] = icu_df['Country Name'].replace('United States', 'US')
    icu_df['Country Name'] = icu_df['Country Name'].replace('Russian Federation', 'Russia')
    icu_df['Country Name'] = icu_df['Country Name'].replace('Iran, Islamic Rep.', 'Iran')
    icu_df['Country Name'] = icu_df['Country Name'].replace('Egypt, Arab Rep.', 'Egypt')
    icu_df['Country Name'] = icu_df['Country Name'].replace('Venezuela, RB', 'Venezuela')

    # We wish to have the most recent values, thus we need to go through every year and extract the most recent one, if it exists.
    icu_cleaned = pd.DataFrame()
    icu_cleaned["country"] = icu_df["Country Name"]
    icu_cleaned["icu"] = np.nan

    for year in range(1960, 2020):
        year_df = icu_df[str(year)].dropna()
        icu_cleaned["icu"].loc[year_df.index] = year_df.values
    return icu_cleaned

import matplotlib.pylab as plt
def show_feature_importance(X,forest):
    """
    Creates a sorted list of the feature importance of a decision tree algorithm.
    Furthermore it plots it.
    params:
        forest: Decision Tree algorithm
    """
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
#     print("Feature ranking:")

#     for f in range(X.shape[1]):
#         print("{}, Feature: {}, Importance: {}".format(f + 1, X.columns[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure(figsize=(10,5))
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices], color="r", align="center")
    plt.xticks(range(X.shape[1]),  X.columns[indices], rotation='vertical')
    plt.xlim([-1, X.shape[1]])
    plt.show()