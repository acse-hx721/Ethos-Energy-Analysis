import pandas as pd

import matplotlib.pyplot as plt
import datetime

# def replace_outliers(df):
#     # Calculate the first and third quartile of each column
#     q1 = df.quantile(0.25)
#     q3 = df.quantile(0.75)
#
#     # Calculate the interquartile range (IQR) of each column
#     iqr = q3 - q1
#
#     # Replace outliers with the maximum value within the acceptable range
#     acceptable_range = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
#     df = df.clip(lower=acceptable_range[0], upper=acceptable_range[1], axis=1)
#
#     return df


def data_preprocessing(df):
    """
        Pre-process the whole data frame
    """
    # delete the replicated index
    df = df.groupby(df.index).last()

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


def simple_estimate_DHW_method1(df, start_date, end_date):
    """
        Method 1 in paper, using minimum value.\n
        Note: Remember to change the date (SH is constant and independent of room temperature)
    """
    # Select a date range that does not use DHW
    # select_date_df = select_rows_by_date(df, 20220701, 20220901)
    select_date_df = select_rows_by_date(df, start_date, end_date)

    # Group the selected time DataFrame by date
    groups = select_date_df.groupby(select_date_df.index.date)
    # find the minimum value of each column
    min_values = groups.min()

    # Make a copy of select_date_df as the result df to avoid warning
    select_date_df_copy = select_date_df.copy()

    # Loop all days to subtract the min value (each day) from each column
    for key, value in groups:
        # print(min_values.loc[key]['ICL.Gas.CHP.1.(Nm3)'])
        # print(select_date_df.loc[value.index]['ICL.Gas.CHP.1.(Nm3)'])
        select_date_df_copy.loc[value.index] = value - min_values.loc[key]
        # print(select_date_df.loc[value.index]['ICL.Gas.CHP.1.(Nm3)'])

    # set negative to 0
    select_date_df_copy[select_date_df_copy < 0] = 0
    # return the updated dataframe
    return select_date_df_copy


def simple_estimate_DHW_method2(df, start_date, end_date, start_time, end_time):
    """
        Method 2 in paper, using average value.\n
        Note: Remember to change the date (SH is constant and independent of room temperature) and time (not use DHW)
    """
    # Select a date range that does not use DHW
    # select_date_df = select_rows_by_date(df, 20220701, 20220901)
    select_date_df = select_rows_by_date(df, start_date, end_date)
    # Select the time range in which the DHW is not used
    select_time_df = select_rows_by_time(select_date_df, start_time, end_time)

    # Group the selected time DataFrame by date
    groups = select_time_df.groupby(select_time_df.index.date)
    # Calculate the average of each column
    mean_values = groups.mean()

    # Make a copy of select_date_df as the result df to avoid warning
    select_date_df_copy = select_date_df.copy()

    # Group the selected date DataFrame by date
    groups_all_day = select_date_df.groupby(select_date_df.index.date)

    # Loop all days to subtract the average value (each day 0:00-4:00) from each column
    for key, value in groups_all_day:
        # print(mean_values.loc[key]['ICL.Gas.CHP.1.(Nm3)'])
        # print(select_date_df.loc[value.index]['ICL.Gas.CHP.1.(Nm3)'])
        select_date_df_copy.loc[value.index] = value - mean_values.loc[key]
        # print(select_date_df.loc[value.index]['ICL.Gas.CHP.1.(Nm3)'])

    # set negative to 0
    select_date_df_copy[select_date_df_copy < 0] = 0
    # return the updated dataframe
    return select_date_df_copy


