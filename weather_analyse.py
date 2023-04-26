#!/usr/bin/env python

# -*- coding:utf-8 -*-
# Author: HongchengXie (Patr1ck)
# Datetime:19/4/2023 下午9:35
# Project Name: PyCharm
from complex_method import *
from utilities import *

# Print Dataframe setting options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Load CSV file into a pandas DataFrame
df_2018_1day = pd.read_csv(DATA_FILE_NAME_2018_1day, parse_dates=[['Date', 'Time']], index_col=0)
df_2022_1day = pd.read_csv(DATA_FILE_NAME_2022_1day, parse_dates=[['Date', 'Time']], index_col=0)

# Pre-processing with nan, negative and the datetime format index
df_2018_1day = data_preprocessing(df_2018_1day, is_1day=True)
df_2022_1day = data_preprocessing(df_2022_1day, is_1day=True)


WEATHER_DATA_FILE_NAME_2018 = 'Weather Data/london 2018-01-01 to 2018-12-31.csv'
WEATHER_DATA_FILE_NAME_2022 = 'Weather Data/london 2022-01-01 to 2022-12-31.csv'

weather_df_2018 = pd.read_csv(WEATHER_DATA_FILE_NAME_2018)
weather_df_2022 = pd.read_csv(WEATHER_DATA_FILE_NAME_2022)

# set the index to the datetime column
weather_df_2018['datetime'] = pd.to_datetime(weather_df_2018['datetime'], format='%Y-%m-%d')
weather_df_2018.set_index('datetime', inplace=True)
weather_df_2022['datetime'] = pd.to_datetime(weather_df_2022['datetime'], format='%Y-%m-%d')
weather_df_2022.set_index('datetime', inplace=True)

# select only the 'temp' columns
weather_df_2018 = weather_df_2018[['temp']]
weather_df_2022 = weather_df_2022[['temp']]

df_combined_2018 = df_2018_1day.join(weather_df_2018, how='left')
df_combined_2022 = df_2022_1day.join(weather_df_2022, how='left')

monthly_plot(df_combined_2018, ['2018 Electricity', 'temperature'],
             title1='2018 Electricity and Temperature Monthly Data',
             title2='2018 Electricity and Temperature Data',
             right_axis='Temperature (°C)',
             y_ticks=np.linspace(0, 4000, num=5),
             y_ticks_2=np.linspace(-10, 40, num=5))

monthly_plot(df_combined_2022, ['2022 Electricity', 'temperature'],
             title1='2022 Electricity and Temperature Monthly Data',
             title2='2022 Electricity and Temperature Data',
             right_axis='Temperature (°C)',
             y_ticks=np.linspace(0, 4000, num=5),
             y_ticks_2=np.linspace(-10, 40, num=5))

# # calculate the correlation coefficient
# corr_coef_2018 = df_combined_2018[DATA_COLUMN_NAME].corr(df_combined_2018['temp'])
# corr_coef_2022 = df_combined_2022[DATA_COLUMN_NAME].corr(df_combined_2022['temp'])

# # Pearson correlation coefficient is a measure of the linear correlation between two variables and ranges from -1 to 1, with 1 indicating a perfect positive correlation, 0 indicating no correlation, and -1 indicating a perfect negative correlation.
# print('Correlation coefficient 2018:', corr_coef_2018)
# print('Correlation coefficient 2022:', corr_coef_2022)
