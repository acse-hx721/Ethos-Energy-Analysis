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


def simple_estimate_DHW_method1(df):
    """
        Method 1 in paper, using minimum value.\n
        Note: Remember to change the date (SH is constant and independent of room temperature)
    """
    # Select a date range that does not use DHW
    # select_date_df = select_rows_by_date(df, 20220701, 20220901)
    select_date_df = select_rows_by_date(df, 20220501, 20220511)
    # find the minimum value of each column
    min_values = select_date_df.min()
    # subtract the minimum value from each column
    df = df.apply(lambda x: x - min_values[x.name])
    # return the updated dataframe
    return df


def simple_estimate_DHW_method2(df):
    """
        Method 2 in paper, using average value.\n
        Note: Remember to change the date (SH is constant and independent of room temperature) and time (not use DHW)
    """
    # Select a date range that does not use DHW
    # select_date_df = select_rows_by_date(df, 20220701, 20220901)
    select_date_df = select_rows_by_date(df, 20220501, 20220511)
    # Select the time range in which the DHW is not used
    select_time_df = select_rows_by_time(select_date_df, '0000', '0400')
    # Calculate the average of each column
    mean_values = select_time_df.mean()
    # subtract the average value from each column
    df = df.apply(lambda x: x - mean_values[x.name])
    # return the updated dataframe
    return df


if __name__ == '__main__':
    # Load CSV file into a pandas DataFrame
    df = pd.read_csv('channelDataExport (5).csv', parse_dates=[['Date', 'Time']], index_col=0)

    # Pre-processing with nan, negative and the datetime format
    df = data_preprocessing(df)

    # print(select_rows_by_date(df, 20220501, 20220503))

    # calculate standard derivation
    # print(df.std())

    # Apply method 1
    result_method1_df = simple_estimate_DHW_method1(df)

    # Apply method 2
    result_method2_df = simple_estimate_DHW_method2(df)

    # Plot the data as a line graph
    df.plot(figsize=(30, 8))

    # Customize the plot
    plt.xlabel('Date Time')
    plt.ylabel('Data Value')
    plt.title('Channel Data')
    plt.legend(title='Channel Names')

    # Display the plot
    plt.show()
