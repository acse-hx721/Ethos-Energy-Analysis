#!/usr/bin/env python

# -*- coding:utf-8 -*-
# Author: HongchengXie (Patr1ck)
# Datetime:24/3/2023 下午2:57
# Project Name: PyCharm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np

# Print Dataframe setting options
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

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
        Front closed and back open intervals\n
        Time string format: HHMM
    """
    # Convert start_time and end_time to string format
    start_str = start_time[:2] + ':' + start_time[2:] + ':00'
    end_str = end_time[:2] + ':' + end_time[2:] + ':00'
    # Create boolean mask for selecting rows within time range
    mask = (df.index.strftime('%H:%M:%S') >= start_str) & (df.index.strftime('%H:%M:%S') < end_str)
    # Select rows within time range
    df_filtered = df.loc[mask]
    return df_filtered


def weekday_weekend_df(df):
    # Monday to Thursday
    weekday_mask = (df.index.weekday >= 0) & (df.index.weekday <= 3)
    # Friday to Sunday
    weekend_mask = (df.index.weekday >= 4) & (df.index.weekday <= 6)
    weekday_df = df.loc[weekday_mask]
    weekend_df = df.loc[weekend_mask]
    return weekday_df, weekend_df

def weekday_weekend_open_close_df(df):
    weekday_df, weekend_df = weekday_weekend_df(df)

    # 2018
    weekday_df_open = select_rows_by_time(weekday_df, '0700', '2200')
    weekend_df_open = select_rows_by_time(weekend_df, '0800', '2000')

    weekday_df_close1 = select_rows_by_time(weekday_df, '2200', '2400')
    weekday_df_close2 = select_rows_by_time(weekday_df, '0000', '0700')
    weekday_df_close = pd.concat([weekday_df_close1, weekday_df_close2], axis=0)

    weekend_df_close1 = select_rows_by_time(weekend_df, '2000', '2400')
    weekend_df_close2 = select_rows_by_time(weekend_df, '0000', '0800')
    weekend_df_close = pd.concat([weekend_df_close1, weekend_df_close2], axis=0)

    weekday_df_close = weekday_df_close.sort_index()
    weekend_df_close = weekend_df_close.sort_index()

    return weekday_df_open, weekday_df_close, weekend_df_open, weekend_df_close


def monthly_plot(df, labels, title1='Monthly Data (separate)', title2='Overall Data', left_axis='Value (kWh)',
                 right_axis='Number of entries'):

    # group the data by month
    grouped = df.groupby(df.index.month)
    # Calculate the maximum and minimum values for all months
    max_val = int(df.max()[df.columns.values[0]] * 1.05)
    min_val = int(df.min()[df.columns.values[0]] * 0.95)

    # Set the y-axis tick locations and labels
    y_ticks = np.linspace(min_val, max_val, num=5)
    if min_val > 100:
        y_tick_labels = [int(y/100)*100 for y in y_ticks]
    else:
        y_tick_labels = [int(y) for y in y_ticks]

    if df.shape[1] == 2:
        # Calculate the maximum and minimum values for all months
        max_val_2 = int(df.max()[df.columns.values[1]] * 1.05)
        min_val_2 = int(df.min()[df.columns.values[1]] * 0.95)
        y_ticks_2 = np.linspace(min_val_2, max_val_2, num=5)
        y_tick_labels_2 = [int(y) for y in y_ticks_2]

    # create subplots for each month's data
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))

    # iterate through each group and plot the data in the corresponding subplot
    for i, ((month, data)) in enumerate(grouped):
        ax = axes[i // 4, i % 4]
        # Plot the first set of data
        data[df.columns.values[0]].plot(ax=ax, color='blue', legend=False)
        ax.set_title('Month {}'.format(month))
        ax.set_xlabel('Date')
        if i % 4 == 0:
            ax.set_ylabel(left_axis, color='blue')
        ax.set_ylim(min_val, max_val)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        ax.tick_params(axis='y', labelcolor='blue')
        if data.shape[1] == 2:
            # Add a second y-axis for the second set of data
            ax2 = ax.twinx()
            # Plot the second set of data
            data[df.columns.values[1]].plot(ax=ax2, color='#FF8C00', legend=False)
            # Set the label for the second y-axis
            if i % 4 == 3:
                ax2.set_ylabel(right_axis, color='#FF8C00')
            # Set the y-axis limits
            ax2.tick_params(axis='y', labelcolor='#FF8C00')
            ax2.set_ylim(min_val_2, max_val_2)

        # Set the x-axis format
        # ax.xaxis.set_major_locator(mdates.MonthLocator())
        # ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
        if month == 2:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    handles, _ = ax.get_legend_handles_labels()
    handles2, _ = ax2.get_legend_handles_labels()
    fig.legend(handles+handles2, labels, loc='lower right')
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle(title1, fontsize=20)
    plt.show()

    # Draw 1 chart for the whole year
    fig, ax = plt.subplots(figsize=(40, 10))
    l1 = ax.plot(df[df.columns.values[0]], color='blue', label=labels[0])
    ax.set_xlabel('Date')
    ax.set_ylabel(left_axis, color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2 = ax.twinx()
    l2 = ax2.plot(df[df.columns.values[1]], color='#FF8C00', label=labels[1])
    ax2.set_ylabel(right_axis, color='#FF8C00')
    ax2.tick_params(axis='y', labelcolor='#FF8C00')
    # set legend
    ax.legend(l1 + l2, labels, loc=0)
    fig.suptitle(title2, fontsize=20)

    # plot every month on x-axis, and set the indicator for each Monday
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
    plt.show()


def monthly_plot_2year(df_2018, df_2022, labels):
    df_2018.index = df_2022.index

    df_concat = pd.concat([df_2018, df_2022])

    # group the data by month
    grouped_concat = df_concat.groupby(df_concat.index.month)

    # Calculate the maximum and minimum values for all months

    max_val = df_concat.max().max() * 1.05
    min_val = df_concat.min().min() * 0.95
    # Set the y-axis tick locations and labels
    y_ticks = np.linspace(min_val, max_val, num=5)
    if min_val > 100:
        y_tick_labels = [int(y/100)*100 for y in y_ticks]
    else:
        y_tick_labels = [int(y) for y in y_ticks]

    # group the data by month
    grouped_2018 = df_2018.groupby(df_2018.index.month)
    grouped_2022 = df_2022.groupby(df_2022.index.month)

    # create subplots for each month's data
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))

    # iterate through each group and plot the data in the corresponding subplot
    for i, ((month, data_2018), (month, data_2022)) in enumerate(zip(grouped_2018, grouped_2022)):
        ax = axes[i // 4, i % 4]
        data_2018.plot(ax=ax, legend=False)
        data_2022.plot(ax=ax, legend=False)
        ax.set_title('Month {}'.format(month))
        ax.set_xlabel('Date')
        ax.set_ylabel('Value(kWh)')
        # Set the x-axis format
        # ax.set_axis_off()
        # ax.xaxis.set_major_locator(mdates.MonthLocator())
        # ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
        # # ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        # if month == 2:
        #     ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        # Set the y-axis limits
        ax.set_ylim(min_val, max_val)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)

    handles, _ = ax.get_legend_handles_labels()
    # print(labels)
    fig.legend(handles, labels, loc='lower center')

    fig.tight_layout()
    plt.show()


def read_visit_file(file_path='Ethos Facility Door access.csv'):
    # Read the CSV file into a dataframe
    df = pd.read_csv(file_path, delimiter=';')
    df_unique_entries = pd.DataFrame(columns=['unique_entry'])


    # Parse dates and set as index
    date_rows = []
    for i, row in df.iterrows():
        try:
            date = pd.to_datetime(row['Row Labels'], format='%d %B %Y')
            date_copy = date
            df_unique_entries.at[i, 'Row Labels'] = date
            df_unique_entries.at[i, 'unique_entry'] = float(row['Sum of daily unique entries'])
            df.drop(index=i, inplace=True)
            # date_rows.append(i)
        except ValueError:
            if row['Row Labels'] == 'Grand Total':
                break
            date_copy = date_copy.replace(hour=int(row['Row Labels']))
            df.at[i, 'Row Labels'] = date_copy

    df.set_index('Row Labels', inplace=True)
    # remove the last row of total entry
    df = df[:-1]
    df = df.rename(columns={'Sum of daily unique entries': 'entries'})
    df_unique_entries.set_index('Row Labels', inplace=True)

    return df, df_unique_entries


def get_weekday_name(number):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return weekdays[number]