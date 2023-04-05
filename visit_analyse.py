#!/usr/bin/env python
import matplotlib.pyplot as plt

# -*- coding:utf-8 -*-
# Author: HongchengXie (Patr1ck)
# Datetime:4/4/2023 上午10:46
# Project Name: PyCharm

from complex_method import *
from utilities import *



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


# monthly_plot(df_combined_2018, ['2018 Electricity', 'Entries'])
# monthly_plot(df_combined_2022, ['2022 Electricity', 'Entries'])

monthly_plot(df_combined_2018_1day, ['2018 Electricity', 'Unique Entries'],
             title1='2018 Electricity and Visitation Monthly Data', title2='2018 Electricity and Visitation Data')
monthly_plot(df_combined_2022_1day, ['2022 Electricity', 'Unique Entries'],
             title1='2022 Electricity and Visitation Monthly Data', title2='2022 Electricity and Visitation Data')

# 统计每周几人数最多 平均值



# 计算两组数据是否存在关联性