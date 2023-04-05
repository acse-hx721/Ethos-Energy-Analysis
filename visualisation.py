#!/usr/bin/env python

# -*- coding:utf-8 -*-
# Author: HongchengXie (Patr1ck)
# Datetime:17/3/2023 下午8:25
# Project Name: PyCharm

import pandas as pd

import matplotlib.pyplot as plt
import datetime



def data_preprocessing(df):
    """
        Pre-process the whole data frame
    """
    # set negative to 0
    df[df < 0] = 0
    # set NaN to 0
    df = df.fillna(0, inplace=False)

    # replace outliers
    # df = replace_outliers(df)
    df[df > 10000] = 0

    # Convert index to datetime format with explicit year component
    # Replace 24:00:00 with 00:00:00 and increment the date by one day
    new_index = pd.to_datetime(df.index.str.replace('24:00:00', '00:00:00'), format='%y/%m/%d %H:%M:%S',
                               yearfirst=True)
    new_index = pd.DatetimeIndex([new_index[i] + pd.offsets.Day(1) if new_index[i].time() == datetime.time(0, 0)
                                  else new_index[i] for i in range(len(new_index))])
    df.index = new_index
    return df


def select_rows_by_date(df, start_date_str, end_date_str):
    """
        Select date range from the 00:00 of start date to 23:30 of (end date - 1)\n
        Date string format: YYYYMMDD
    """
    # Convert start and end date strings to datetime objects
    start_date = pd.to_datetime(start_date_str, format='%Y%m%d')
    end_date = pd.to_datetime(end_date_str, format='%Y%m%d')
    # Create boolean mask for selecting rows within date range
    # Change if include the end date or not
    mask = (df.index >= start_date) & (df.index < end_date)
    # Select rows within date range
    df_filtered = df.loc[mask]
    return df_filtered


def select_rows_by_time(df, start_time, end_time):
    """
        Select time range from start time to end time\n
        Time string format: HHMM
    """
    # Convert start_time and end_time to string format
    start_str = start_time[:2] + ':' + start_time[2:] + ':00'
    end_str = end_time[:2] + ':' + end_time[2:] + ':00'
    # Create boolean mask for selecting rows within time range
    mask = (df.index.strftime('%H:%M:%S') >= start_str) & (df.index.strftime('%H:%M:%S') <= end_str)
    # Select rows within time range
    df_filtered = df[mask]
    return df_filtered


def plot_df_chart(df, title):
    # Plot the result df as line graph
    df.plot(figsize=(30, 8))
    plt.title(title)
    # Customize the plot
    plt.xlabel('Date Time')
    plt.ylabel('Data Value')
    plt.legend(title='Channel Names')

    # Display the plot
    plt.show()



if __name__ == '__main__':
    # Load CSV file into a pandas DataFrame
    # df = pd.read_csv('channelDataExport (5).csv', parse_dates=[['Date', 'Time']], index_col=0)

    df_water_2018 = pd.read_csv('Ethos_sigma_data_2018&2022/ethos_water/water_2018_1day.csv', parse_dates=[['Date', 'Time']],
                                index_col=0)
    df_water_2022 = pd.read_csv('Ethos_sigma_data_2018&2022/ethos_water/water_2022_1day.csv', parse_dates=[['Date', 'Time']],
                                index_col=0)

    df_heat_2018 = pd.read_csv('Ethos_sigma_data_2018&2022/ethos_heat/heat_2018_1h.csv', parse_dates=[['Date', 'Time']],
                               index_col=0)
    df_heat_2022 = pd.read_csv('Ethos_sigma_data_2018&2022/ethos_heat/heat_2022_1h.csv', parse_dates=[['Date', 'Time']],
                               index_col=0)

    df_elec_2018 = pd.read_csv('Ethos_sigma_data_2018&2022/ethos_electricity/elec_2018_1h.csv',
                               parse_dates=[['Date', 'Time']],
                               index_col=0)
    df_elec_2022 = pd.read_csv('Ethos_sigma_data_2018&2022/ethos_electricity/elec_2022_1h.csv',
                               parse_dates=[['Date', 'Time']],
                               index_col=0)

    # Pre-processing with nan, negative and the datetime format
    df_water_2018 = data_preprocessing(df_water_2018)
    df_water_2022 = data_preprocessing(df_water_2022)

    df_heat_2018 = data_preprocessing(df_heat_2018)
    df_heat_2022 = data_preprocessing(df_heat_2022)

    df_elec_2018 = data_preprocessing(df_elec_2018)
    df_elec_2022 = data_preprocessing(df_elec_2022)

    # Plot the result df as line graph
    plot_df_chart(df_water_2018, 'Water 2018 - From Sigma - 1 day')
    plot_df_chart(df_water_2022, 'Water 2022 - From Sigma - 1 day')

    plot_df_chart(df_heat_2018, 'Heat 2018 - From Sigma - 1 h')
    plot_df_chart(df_heat_2022, 'Heat 2022 - From Sigma - 1 h')

    plot_df_chart(df_elec_2018, 'Electricity 2018 - From Sigma - 1 h')
    plot_df_chart(df_elec_2022, 'Electricity 2022 - From Sigma - 1 h')
