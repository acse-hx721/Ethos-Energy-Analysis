#!/usr/bin/env python
import matplotlib.pyplot as plt

# -*- coding:utf-8 -*-
# Author: HongchengXie (Patr1ck)
# Datetime:4/4/2023 上午10:46
# Project Name: PyCharm

from complex_method import *
from utilities import *

# Print Dataframe setting options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load CSV file into a pandas DataFrame
df_2018 = pd.read_csv(DATA_FILE_NAME_2018, parse_dates=[['Date', 'Time']], index_col=0)
df_2022 = pd.read_csv(DATA_FILE_NAME_2022, parse_dates=[['Date', 'Time']], index_col=0)

df_2018_1day = pd.read_csv(DATA_FILE_NAME_2018_1day, parse_dates=[['Date', 'Time']], index_col=0)
df_2022_1day = pd.read_csv(DATA_FILE_NAME_2022_1day, parse_dates=[['Date', 'Time']], index_col=0)

# Pre-processing with nan, negative and the datetime format index
df_2018 = data_preprocessing(df_2018)
df_2022 = data_preprocessing(df_2022)

# Some magic steps for data (when interval != 1day)...
df_2018 = df_2018.iloc[:-1]
df_2022 = df_2022.iloc[:-1]

# Pre-processing with nan, negative and the datetime format index
df_2018_1day = data_preprocessing(df_2018_1day, is_1day=True)
df_2022_1day = data_preprocessing(df_2022_1day, is_1day=True)


# read the file of visitation
df_entries, df_unique_entries = read_visit_file()

# Combine the two DataFrames using join()
df_combined_2018 = df_2018.join(df_entries, how='left')
df_combined_2022 = df_2022.join(df_entries, how='left')

df_combined_2018_1day = df_2018_1day.join(df_unique_entries, how='left')
df_combined_2022_1day = df_2022_1day.join(df_unique_entries, how='left')

# Set the index of the combined DataFrame to be the index of df1
df_combined_2018.index = df_2018.index
df_combined_2022.index = df_2022.index

df_combined_2018_1day.index = df_2018_1day.index
df_combined_2022_1day.index = df_2022_1day.index

# print(df_combined_2018)
df_combined_2018_1day[['unique_entry']] = df_combined_2018_1day[['unique_entry']].fillna(0)
df_combined_2022_1day[['unique_entry']] = df_combined_2022_1day[['unique_entry']].fillna(0)

# monthly_plot(df_combined_2018, ['2018 Electricity', 'Entries'])
# monthly_plot(df_combined_2022, ['2022 Electricity', 'Entries'])

monthly_plot(df_combined_2018_1day, ['2018 Electricity', 'Unique Entries'],
             title1='2018 Electricity and Visitation Monthly Data', title2='2018 Electricity and Visitation Data')
monthly_plot(df_combined_2022_1day, ['2022 Electricity', 'Unique Entries'],
             title1='2022 Electricity and Visitation Monthly Data', title2='2022 Electricity and Visitation Data')

# # 统计每周几人数最多 平均值
# print(df_combined_2018_1day.index.weekday)
#
# 根据周几分groups
groups_week_2018 = df_combined_2018_1day.groupby(df_combined_2018_1day.index.weekday)
df_weekday_pattern_2018 = pd.DataFrame(columns=['Weekday', '2018 Average Consumption', '2018 Average Visitation'])

groups_week_2022 = df_combined_2022_1day.groupby(df_combined_2022_1day.index.weekday)
df_weekday_pattern_2022 = pd.DataFrame(columns=['Weekday', '2022 Average Consumption', '2022 Average Visitation'])

for weekday, group in groups_week_2018:
    df_weekday_pattern_2018 = df_weekday_pattern_2018.append({'Weekday': get_weekday_name(weekday),
                                                              '2018 Average Consumption': group.mean()[0],
                                                              '2018 Average Visitation': group.mean()[1]},
                                                             ignore_index=True)
    # print(weekday)
    # print(group)
    # print(group.mean())

for weekday, group in groups_week_2022:
    df_weekday_pattern_2022 = df_weekday_pattern_2022.append({'Weekday': get_weekday_name(weekday),
                                                              '2022 Average Consumption': group.mean()[0],
                                                              '2022 Average Visitation': group.mean()[1]},
                                                             ignore_index=True)


# Weekly pattern
df_weekday_pattern_2018 = df_weekday_pattern_2018.set_index(['Weekday']).round(2)
df_weekday_pattern_2022 = df_weekday_pattern_2022.set_index(['Weekday']).round(2)

df_weekly_pattern_combine = pd.concat([df_weekday_pattern_2018, df_weekday_pattern_2022], axis=1, join='inner')
# print(df_weekly_pattern_combine)
# save the dataframe as a CSV file
# df_weekly_pattern_combine.to_csv('df_weekly_pattern_combine.csv')
# save the dataframe as an Excel file
# df_weekly_pattern_combine.to_excel('df_weekly_pattern_combine.xlsx')


# Daily pattern
# Define start and end times for each period

periods = {
    'morning': ['0700', '1200'],
    'afternoon': ['1200', '1800'],
    'evening': ['1800', '2200']
}

# df_combined_2018_morning = select_rows_by_time(df_combined_2018, periods['morning'][0], periods['morning'][1])
# df_combined_2018_afternoon = select_rows_by_time(df_combined_2018, periods['afternoon'][0], periods['afternoon'][1])
# df_combined_2018_evening = select_rows_by_time(df_combined_2018, periods['evening'][0], periods['evening'][1])
#
#
# df_combined_2022_morning = select_rows_by_time(df_combined_2022, periods['morning'][0], periods['morning'][1])
# df_combined_2022_afternoon = select_rows_by_time(df_combined_2022, periods['afternoon'][0], periods['afternoon'][1])
# df_combined_2022_evening = select_rows_by_time(df_combined_2022, periods['evening'][0], periods['evening'][1])

# 全天开放时间的分析 7am - 9pm
df_daily_pattern_2018 = pd.DataFrame(columns=['Time', '2018 Average Consumption', '2018 Average Visitation'])
df_daily_pattern_2022 = pd.DataFrame(columns=['Time', '2022 Average Consumption', '2022 Average Visitation'])

df_combined_2018_open = select_rows_by_time(df_combined_2018, periods['morning'][0], periods['evening'][1])
df_combined_2018_open_group = df_combined_2018_open.groupby(df_combined_2018_open.index.time)

df_combined_2022_open = select_rows_by_time(df_combined_2022, periods['morning'][0], periods['evening'][1])
df_combined_2022_open_group = df_combined_2022_open.groupby(df_combined_2022_open.index.time)

for i, group in df_combined_2018_open_group:
    df_daily_pattern_2018 = df_daily_pattern_2018.append({'Time': str(i),
                                                          '2018 Average Consumption': group.mean()[0],
                                                          '2018 Average Visitation': group.mean()[1]},
                                                         ignore_index=True)
for i, group in df_combined_2022_open_group:
    df_daily_pattern_2022 = df_daily_pattern_2022.append({'Time': str(i),
                                                          '2022 Average Consumption': group.mean()[0],
                                                          '2022 Average Visitation': group.mean()[1]},
                                                         ignore_index=True)

# Set index and round to 2 decimal
df_daily_pattern_2018 = df_daily_pattern_2018.set_index(['Time']).round(2)
df_daily_pattern_2022 = df_daily_pattern_2022.set_index(['Time']).round(2)

df_daily_pattern_combine = pd.concat([df_daily_pattern_2018, df_daily_pattern_2022], axis=1, join='inner')

# print(df_daily_pattern_combine)

# save the dataframe as a CSV file
# df_daily_pattern_combine.to_csv('df_daily_pattern_combine.csv')
# save the dataframe as an Excel file
# df_daily_pattern_combine.to_excel('df_daily_pattern_combine.xlsx')

# print(df_combined_2018_1day)



# calculate the correlation coefficient
corr_coef_2018 = df_combined_2018_1day[DATA_COLUMN_NAME].astype('float64').corr(df_combined_2018_1day['unique_entry'].astype('float64'))
corr_coef_2022 = df_combined_2022_1day[DATA_COLUMN_NAME].astype('float64').corr(df_combined_2022_1day['unique_entry'].astype('float64'))

# Pearson correlation coefficient is a measure of the linear correlation between two variables and ranges from -1 to 1, with 1 indicating a perfect positive correlation, 0 indicating no correlation, and -1 indicating a perfect negative correlation.
print('Correlation coefficient 2018:', corr_coef_2018)
print('Correlation coefficient 2022:', corr_coef_2022)
