#!/usr/bin/env python

# -*- coding:utf-8 -*-
# Author: HongchengXie (Patr1ck)
# Datetime:4/4/2023 上午10:54
# Project Name: PyCharm

from complex_method import *
from utilities import *


# main_process(DATA_FILE_NAME_2018)
# main_process(DATA_FILE_NAME_2022)

# Load CSV file into a pandas DataFrame
df_2018 = pd.read_csv(DATA_FILE_NAME_2018, parse_dates=[['Date', 'Time']], index_col=0)
df_2022 = pd.read_csv(DATA_FILE_NAME_2022, parse_dates=[['Date', 'Time']], index_col=0)

# Pre-processing with nan, negative and the datetime format index
df_2018 = data_preprocessing(df_2018)
df_2022 = data_preprocessing(df_2022)

# Some magic steps...
df_2018 = df_2018.iloc[:-1]
df_2022 = df_2022.iloc[:-1]

df_2018_index = df_2018.index
df_2018.index = df_2022.index #need to be deleted

# df_2022.index = df_2022.index.to_series().apply(lambda x: x.strftime('%m-%d %H:%M:%S'))

# group the data by month
grouped_2018 = df_2018.groupby(df_2018.index.month)
grouped_2022 = df_2022.groupby(df_2022.index.month)

sum_month_2018 = grouped_2018.sum()
sum_month_2018 = sum_month_2018.rename(columns={DATA_COLUMN_NAME: '2018'})
sum_month_2022 = grouped_2022.sum()
sum_month_2022 = sum_month_2022.rename(columns={DATA_COLUMN_NAME: '2022'})
result_month_sum = pd.concat([sum_month_2018, sum_month_2022], axis=1)
print(result_month_sum)

quarter_df = pd.DataFrame({'2018': [float(sum_month_2018.iloc[0:3].sum()), float(sum_month_2018.iloc[3:6].sum()),
                                    float(sum_month_2018.iloc[6:9].sum()), float(sum_month_2018.iloc[9:12].sum())],
                           '2022': [float(sum_month_2022.iloc[0:3].sum()), float(sum_month_2022.iloc[3:6].sum()),
                                    float(sum_month_2022.iloc[6:9].sum()), float(sum_month_2022.iloc[9:12].sum())]})
quarter_df.index = ['Q1', 'Q2', 'Q3', 'Q4']
print(quarter_df)

monthly_plot_2year(df_2018, df_2022, ['2018', '2022'])


# restore the index of 2018 dataframe
df_2018.index = df_2018_index
monthly_plot(df_2018, ['2018'])
monthly_plot(df_2022, ['2022'])

weekday_df_2018_open, weekday_df_2018_close, weekend_df_2018_open, weekend_df_2018_close = \
    weekday_weekend_open_close_df(df_2018)

weekday_df_2022_open, weekday_df_2022_close, weekend_df_2022_open, weekend_df_2022_close = \
    weekday_weekend_open_close_df(df_2022)

week_df = pd.DataFrame({'2018': [float(weekday_df_2018_open.mean()), float(weekday_df_2018_close.mean()),
                                 float(weekend_df_2018_open.mean()), float(weekend_df_2018_close.mean())],
                        '2022': [float(weekday_df_2022_open.mean()), float(weekday_df_2022_close.mean()),
                                 float(weekend_df_2022_open.mean()), float(weekend_df_2022_close.mean())]})
week_df.index = ['Weekday open (0700-2200)', 'Weekday close', 'Weekend open (0800-2000)', 'Weekend close']
print(week_df)

df_2018_result = separate_and_estimate_ETHOS_open_close(df_2018)
df_2022_result = separate_and_estimate_ETHOS_open_close(df_2022)

df_2018_result_drop = df_2018_result.drop(labels=DATA_COLUMN_NAME, axis=1)
df_2022_result_drop = df_2022_result.drop(labels=DATA_COLUMN_NAME, axis=1)

monthly_plot_2year(df_2018_result_drop, df_2022_result_drop, ['2018 base', '2018 other', '2022 base', '2022 other'])

calculation_df = pd.DataFrame({'2018': [float(df_2018_result_drop['base'].mean()), float(df_2018_result_drop['other'].mean())],
                        '2022': [float(df_2022_result_drop['base'].mean()), float(df_2022_result_drop['other'].mean())]})
calculation_df.index = ['Base consumption', 'Other consumption']
print(calculation_df)
