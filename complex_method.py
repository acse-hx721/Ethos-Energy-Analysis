import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import datetime
# import scipy
from scipy import stats
# from scipy import interpolate
# from scipy.interpolate import StinemanInterpolator

# from statsmodels.tsa.statespace.kalman_filter import (
#     KalmanFilter,
#     FilterResults,
# )

from pykalman import KalmanFilter
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR

from utilities import *

import statsmodels.api as sm


# Time interval of date, unit is minutes
TIME_INTERVAL = 30

# File in different interval, remember to change
DATA_FILE_NAME_2018 = 'Ethos_sigma_data_2018&2022/ethos_electricity/elec_2018_1h.csv'
DATA_FILE_NAME_2022 = 'Ethos_sigma_data_2018&2022/ethos_electricity/elec_2022_1h.csv'
DATA_FILE_NAME_2018_1day = 'Ethos_sigma_data_2018&2022/ethos_electricity/elec_2018_1day.csv'
DATA_FILE_NAME_2022_1day = 'Ethos_sigma_data_2018&2022/ethos_electricity/elec_2022_1day.csv'


DATA_COLUMN_NAME = 'sk-nor-101-sip1.ad.ic.ac.uk_Device_1'

WEATHER_DATA_FILE_NAME = 'london_weather2022_fake.csv'

# Print Dataframe setting options
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)


def replace_outliers(df):
    # Calculate the first and third quartile of each column
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)

    # Calculate the interquartile range (IQR) of each column
    iqr = q3 - q1

    # Replace outliers with the maximum value within the acceptable range
    acceptable_range = (0, q3 + 1.5 * iqr)
    # print(df)
    # for i in range(1, len(df)):
    #     if df.iloc[i, 0] > acceptable_range[1]:
    #         df.iloc[i, 0] = df.iloc[i - 1, 0]
    # df = df.clip(lower=acceptable_range[0], upper=acceptable_range[1], axis=1)

    # Replace outlier values with previous value
    print(df)
    df[df > acceptable_range[1]] = np.nan
    # df.loc[df[DATA_COLUMN_NAME] > acceptable_range[1], DATA_COLUMN_NAME] = np.nan
    df = df.fillna(method='ffill')

    return df


def data_preprocessing(df, is_1day=False):
    """
        Pre-process the whole data frame\n
        Remember 1 day interval data have different process
    """
    # set negative to 0
    df[df < 0] = 0
    # set NaN to 0
    df = df.fillna(0, inplace=False)

    # replace outliers
    df = replace_outliers(df)

    # df[df == 0] = np.NaN

    # Convert index to datetime format with explicit year component
    # Replace 24:00:00 with 00:00:00 and increment the date by one day
    new_index = pd.to_datetime(df.index.str.replace('24:00:00', '00:00:00'), format='%y/%m/%d %H:%M:%S',
                               yearfirst=True)
    if not is_1day:
        new_index = pd.DatetimeIndex([new_index[i] + pd.offsets.Day(1) if new_index[i].time() == datetime.time(0, 0)
                                     else new_index[i] for i in range(len(new_index))])
    df.index = new_index
    return df





# ================================== FIND DHW+SH NA POINTS METHODS==============================
def maximum_peaks_find(df, num_peaks=7, start_time='0500', end_time='2400'):

    # Select the time range in which the DHW_SH is used
    select_time_df = select_rows_by_time(df, start_time, end_time)

    # Group the selected time DataFrame by date
    groups = select_time_df.groupby(select_time_df.index.date)

    # Loop through each group and set the "is_na" attribute to True for the largest 7 data points
    for date, group in groups:
        largest_n = group.nlargest(num_peaks, DATA_COLUMN_NAME)
        select_time_df.loc[largest_n.index, 'is_NA'] = True

    # print(select_time_df[:50])
    result_df = pd.concat([select_time_df, df], axis=0)
    result_df = result_df[~result_df.index.duplicated(keep='first')]
    result_df = result_df.sort_index()

    return result_df


def expected_profiles_find(df):
    # Split dataframe into two based on day of the week # Monday to Thursday
    # Monday to Thursday
    weekday_mask = (df.index.weekday >= 0) & (df.index.weekday <= 3)
    # Friday to Sunday
    weekend_mask = (df.index.weekday >= 4) & (df.index.weekday <= 6)
    weekday_df = df.loc[weekday_mask]
    weekend_df = df.loc[weekend_mask]

    # print(weekday_df[:100])
    # print(weekend_df[:100])

    # display the combined data frame
    # print(combined_df)

    weekend_df_NA = maximum_peaks_find(weekend_df)

    result_df = weekend_df_NA.copy()

    # Group the selected time DataFrame by date
    groups = weekday_df.groupby(weekday_df.index.date)

    # Define start and end times for each period
    periods = {
        'morning': ['0500', '1100'],
        'afternoon': ['1200', '1600'],
        'evening': ['1700', '2359']
    }

    # Define function to mark rows as NA
    def mark_as_na(df, start_time, end_time, time_interval=TIME_INTERVAL):
        max_value_index = df[DATA_COLUMN_NAME].idxmax()
        prev_time = max_value_index - pd.Timedelta(minutes=time_interval)
        if prev_time >= df.index[0]:
            # Mark previous row as NA if it's within the same period
            # if prev_time.strftime('%H:%M:%S') >= start_time and prev_time.strftime('%H:%M:%S') <= end_time:
            df.loc[prev_time, 'is_NA'] = True
        next_time = max_value_index + pd.Timedelta(minutes=time_interval)
        if next_time <= df.index[-1]:
            # Mark next row as NA if it's within the same period
            # if next_time.strftime('%H:%M:%S') >= start_time and next_time.strftime('%H:%M:%S') <= end_time:
            df.loc[next_time, 'is_NA'] = True
        return df

    # Iterate over each day and each period, and apply the select_rows_by_time function
    for name, group in groups:
        for period, (start_time, end_time) in periods.items():
            period_df = select_rows_by_time(group, start_time, end_time)
            if not period_df.empty:
                # Mark largest value row as NA
                max_value_index = period_df[DATA_COLUMN_NAME].idxmax()
                period_df.loc[max_value_index, 'is_NA'] = True
                if period == 'morning' or period == 'evening':
                    # Mark adjacent rows as NA for morning and evening periods
                    period_df = mark_as_na(period_df, start_time, end_time)
                # print(f'{name}: {period} - {period_df}')
                # print(period_df)
                # concatenate the two data frames
                result_df = pd.concat([result_df, period_df], axis=0)

    result_df = pd.concat([result_df, weekday_df], axis=0)
    result_df = result_df[~result_df.index.duplicated(keep='first')]
    # sort the index to match the original order
    result_df = result_df.sort_index()
    return result_df


def outdoor_temperature_find(df):
    weather_df = pd.read_csv(WEATHER_DATA_FILE_NAME, index_col=0)
    new_index = pd.to_datetime(weather_df.index, format='%Y%m%d', yearfirst=True)
    weather_df.index = new_index

    # 1. HEATING SEASON CALCULATION
    heating_season_df1 = select_rows_by_date(df, '20220101', '20220601')
    heating_season_df2 = select_rows_by_date(df, '20220901', '20230101')

    heating_season_df = pd.concat([heating_season_df1, heating_season_df2], axis=0)
    heating_season_selected_time_df = select_rows_by_time(heating_season_df, '0100', '0400')

    # Group the selected time DataFrame by date, and calculate the average of each day
    heating_season_selected_time_group = heating_season_selected_time_df.groupby(
                                         heating_season_selected_time_df.index.date)
    heating_season_selected_time_average = heating_season_selected_time_group[DATA_COLUMN_NAME].mean()

    # Select the weather of the heating season, Remember to check the nan of weather series
    heating_season_weather_series = weather_df.loc[heating_season_selected_time_average.index]['mean_temp']

    # Perform linear regression between daily demand and the average temperature
    heating_season_slope, heating_season_intercept, _, _, _ = stats.linregress(heating_season_selected_time_average,
                                                                               heating_season_weather_series)
    # print(heating_season_slope, heating_season_intercept)

    # Create a function to predict heating demand from daily temperature
    def heating_season_predict_demand(daily_temp):
        return heating_season_slope * daily_temp + heating_season_intercept + 0.9

    # 2. NO HEATING SEASON CALCULATION
    no_heating_season_df = select_rows_by_date(df, '20220601', '20220901')
    no_heating_season_selected_time_df = select_rows_by_time(no_heating_season_df, '0100', '0400')

    # Group the selected time DataFrame by date, and calculate the average of each day
    no_heating_season_selected_time_group = no_heating_season_selected_time_df.groupby(
                                            no_heating_season_selected_time_df.index.date)
    no_heating_season_selected_time_average = no_heating_season_selected_time_group[DATA_COLUMN_NAME].mean()

    # Select the weather of the no heating season, Remember to check the nan of weather series
    no_heating_season_weather_series = weather_df.loc[no_heating_season_selected_time_average.index]['mean_temp']
    # print(no_heating_season_weather_df)

    # Perform linear regression between daily demand and the average temperature
    no_heating_season_slope, no_heating_season_intercept, _, _, _ = stats.linregress(
                                                                    no_heating_season_selected_time_average,
                                                                    no_heating_season_weather_series)
    # print(no_heating_season_slope, no_heating_season_intercept)

    # Create a function to predict heating demand from daily temperature
    def no_heating_season_predict_demand(daily_temp):
        return no_heating_season_slope * daily_temp + no_heating_season_intercept + 0.6

    # Create an empty DataFrame to store the results
    result_df = pd.DataFrame(columns=df.columns)

    # SET HEATING SEASON NA value
    heating_season_group = heating_season_df.groupby(heating_season_df.index.date)
    for date, group in heating_season_group:
        # Combine with a time object to create a datetime.datetime object 2022-05-01 to 2022-05-01 00:00:00
        datetime_type_date = datetime.datetime.combine(date, datetime.time.min)
        # Calculate the daily average temperature
        daily_temp = heating_season_weather_series[datetime_type_date]
        # Compare the actual value with the predicted value and mark as NA if actual > predicted
        group.loc[group[DATA_COLUMN_NAME] > heating_season_predict_demand(daily_temp), 'is_NA'] = True
        # Append the marked group to the results DataFrame
        result_df = pd.concat([result_df, group])

    # SET NO HEATING SEASON NA value
    no_heating_season_group = no_heating_season_df.groupby(no_heating_season_df.index.date)
    for date, group in no_heating_season_group:
        # Combine with a time object to create a datetime.datetime object 2022-05-01 to 2022-05-01 00:00:00
        datetime_type_date = datetime.datetime.combine(date, datetime.time.min)
        # Calculate the daily average temperature
        daily_temp = no_heating_season_weather_series[datetime_type_date]
        # Compare the actual value with the predicted value and mark as NA if actual > predicted
        group.loc[group[DATA_COLUMN_NAME] > no_heating_season_predict_demand(daily_temp), 'is_NA'] = True
        # Append the marked group to the results DataFrame
        result_df = pd.concat([result_df, group])

    # sort the index to match the original order
    result_df = result_df.sort_index()
    return result_df


def combine_find_method_1(df):
    """
    Combine method 1
    Only categorizes a data point if both the maximum_peaks_find() and outdoor_temperature_find()
    label the same point as NA value
    """
    maximum_peaks_find_df = maximum_peaks_find(df)
    outdoor_temperature_find_df = outdoor_temperature_find(df)

    result_df = maximum_peaks_find_df.copy()

    # Combine the two dataframes on the index
    combined_series = maximum_peaks_find_df['is_NA'].combine(outdoor_temperature_find_df['is_NA'], lambda x, y: x and y)
    result_df['is_NA'] = combined_series

    print(maximum_peaks_find_df)
    print("========================================================================================================================")
    print(outdoor_temperature_find_df)
    print("========================================================================================================================")
    print(result_df)

    return result_df


def combine_find_method_2(df):
    """
    Combine method 2:
    If the total energy of the datapoint is lower than 250 Wh or higher than 3,000 Wh,
    then the “Outdoor temperature approach” is used. If not, the “Maximum peaks approach” is used.
    """
    ...


# ================================== ## Calculate the DHW from NA value METHODS ==============================
def interpolation_estimate_method(df, method='linear'):
    estimate_df = df.copy()
    estimate_df['SH'] = estimate_df[DATA_COLUMN_NAME]
    # Create a boolean mask for identifying rows with is_NA = True
    mask = estimate_df['is_NA'] == True
    # Create a new column
    estimate_df.loc[mask, 'SH'] = np.nan
    # Interpolate missing values using linear interpolation
    if method == "linear":
        estimate_df['SH'] = estimate_df['SH'].interpolate(method='linear')
    elif method == "cubic spline":
        estimate_df['SH'] = estimate_df['SH'].interpolate(method='spline', order=3)
    # elif method == "Stineman":
    #     # Create a StinemanInterpolator object
    #     interp = StinemanInterpolator(estimate_df.index, estimate_df['SH'])
    #     # Interpolate missing values using Stineman interpolation
    #     estimate_df['SH'] = interp(estimate_df.index)

    # Calculate water heating use for rows with is_NA = True
    estimate_df.loc[mask, 'DHW'] = estimate_df.loc[mask, DATA_COLUMN_NAME] - estimate_df.loc[mask, 'SH']

    return estimate_df


def moving_average_estimate_method(df, average_method='SMA'):
    # Define the window sizes and weights for the three moving averages
    simple_window = 4
    linear_window = 2
    linear_weights = [1 / 3, 1 / 2, 0, 1 / 2, 1 / 3]
    exp_window = 2
    exp_weights = [1 / 4, 1 / 2, 0, 1 / 2, 1 / 4]

    estimate_df = df.copy()
    # Calculate water heating use for rows with is_NA = True
    mask = estimate_df['is_NA'] == True
    # Create a new column
    estimate_df.loc[mask, 'SH'] = np.nan

    # Select different average methods
    if average_method == 'SMA':
        # Calculate the simple moving average
        estimate_df['SMA'] = estimate_df[DATA_COLUMN_NAME].rolling(window=simple_window, min_periods=1).mean()
        estimate_df['SH'] = estimate_df.apply(lambda row: row['SMA'] if row['is_NA'] else row[DATA_COLUMN_NAME]
                                              , axis=1)
    elif average_method == 'LMA':
        # Calculate the linear weighted moving average
        estimate_df['LMA'] = estimate_df[DATA_COLUMN_NAME].rolling(window=linear_window * 2 + 1, min_periods=5,
                                                                        center=True) \
            .apply(lambda x: np.average(x, weights=linear_weights, axis=0))
        estimate_df['SH'] = estimate_df.apply(lambda row: row['LMA'] if row['is_NA'] else row[DATA_COLUMN_NAME]
                                              , axis=1)
    elif average_method == 'EMA':
        # Calculate the exponential weighted moving average
        estimate_df['EMA'] = estimate_df[DATA_COLUMN_NAME].rolling(window=exp_window * 2 + 1, min_periods=5,
                                                                        center=True) \
            .apply(lambda x: np.average(x, weights=exp_weights, axis=0))
        estimate_df['SH'] = estimate_df.apply(lambda row: row['EMA'] if row['is_NA'] else row[DATA_COLUMN_NAME]
                                              , axis=1)
    # estimate_df['DHW'] = estimate_df.apply(lambda row: row['EMA'] - row['SH'] if row['is_NA'] else 0, axis=1)

    estimate_df.loc[mask, 'DHW'] = estimate_df.loc[mask, DATA_COLUMN_NAME] - estimate_df.loc[mask, 'SH']
    estimate_df_drop = estimate_df.drop(labels=average_method, axis=1)

    return estimate_df_drop


def kalman_filtering_estimate_method_old(df):
    def kalman_filter(data, model_type="structural_smooth"):
        """
        Applies a Kalman filter to a given dataframe of time series data.

        Args:
            data (pandas.DataFrame): The time series data to filter.
            model_type (str): The type of model to use for the filter.
                Options are "structural_smooth", "structural_nosmooth",
                "arima_smooth", or "arima_nosmooth". Default is
                "structural_smooth".

        Returns:
            pandas.DataFrame: The filtered time series data.
        """
        # Define the state space model for the Kalman filter
        if model_type == "structural_smooth":
            model = UnobservedComponents(
                data[DATA_COLUMN_NAME],
                level="local level",
                seasonal=12,
                cycle=False,
                stochastic_level=True,
                damped_cycle=False,
            )
        elif model_type == "structural_nosmooth":
            model = UnobservedComponents(
                data[DATA_COLUMN_NAME],
                level="local level",
                seasonal=12,
                cycle=False,
                stochastic_level=False,
                damped_cycle=False,
            )
        elif model_type == "arima_smooth":
            model = ARIMA(data[DATA_COLUMN_NAME], order=(1, 1, 1))
        elif model_type == "arima_nosmooth":
            model = ARIMA(data[DATA_COLUMN_NAME], order=(1, 1, 1), trend="nc")
        else:
            raise ValueError("Invalid model type.")

        # Fit the model and run the Kalman filter
        results = model.fit()
        k_endog = data[DATA_COLUMN_NAME].values  # observing only one variable
        kf = KalmanFilter(
            k_endog=k_endog,
            k_states=model.k_states,
            model=model,
            results=results,
            loglikelihood_burn=results.loglikelihood_burn,
            save_filtered=True,
            save_smoothed=True,
        )
        # kf.fit()
        # kf.initialize_known(data[DATA_COLUMN_NAME].values)
        # bind the data to the Kalman filter object
        endog = data[DATA_COLUMN_NAME].values.reshape(1, -1)[0]
        kf.bind(endog)
        # data[DATA_COLUMN_NAME] = data[DATA_COLUMN_NAME].astype(int)

        # print(len(endog))
        kf_results = kf.filter(endog)

        # Add the filtered data to the dataframe
        data["filtered"] = kf_results.filtered_state[0]
        data["smoothed"] = kf_results.smoothed_state[0]

        return data

    # Load the data
    estimate_df = df.copy()

    # Calculate the space heating use and water heating use
    estimate_df["SH"] = estimate_df[DATA_COLUMN_NAME]
    estimate_df.loc[estimate_df["is_NA"], "SH"] = np.nan
    estimate_df = kalman_filter(estimate_df, model_type="structural_smooth")
    estimate_df["SH"] = estimate_df["filtered"]
    estimate_df["DHW"] = estimate_df[DATA_COLUMN_NAME] - estimate_df["SH"]

    return estimate_df


def kalman_filtering_estimate_method(df, model_type):

    # Define the Kalman filter models
    # Model 1: Structural time series model with smoothing
    model1 = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        transition_matrices=np.array([[1, 1], [0, 1]]),
        observation_matrices=np.array([[1, 0]]),
        initial_state_mean=np.array([df[DATA_COLUMN_NAME].iloc[0], 0]),
        initial_state_covariance=np.diag([1, 1]),
        observation_covariance=1,
        transition_covariance=np.diag([0.1, 0.1]),
        em_vars=["transition_covariance", "observation_covariance", "initial_state_covariance"]
    )

    # Model 2: Structural time series model without smoothing
    model2 = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        transition_matrices=np.array([[1, 1], [0, 1]]),
        observation_matrices=np.array([[1, 0]]),
        initial_state_mean=np.array([df[DATA_COLUMN_NAME].iloc[0], 0]),
        initial_state_covariance=np.diag([1, 1]),
        observation_covariance=1,
        transition_covariance=np.diag([0.1, 0.1]),
        em_vars=[]
    )

    # # Model 3: ARIMA model with smoothing
    # model3 = KalmanFilter(
    #     n_dim_obs=1,
    #     n_dim_state=2,
    #     transition_matrices=np.array([[1, 1], [0, 1]]),
    #     observation_matrices=np.array([[1, 0]]),
    #     initial_state_mean=np.array([df[DATA_COLUMN_NAME].iloc[0], 0]),
    #     initial_state_covariance=np.diag([1, 1]),
    #     observation_covariance=1,
    #     transition_covariance=np.diag([0.1, 0.1]),
    #     em_vars=["transition_covariance", "observation_covariance", "initial_state_covariance"],
    #     observation_offsets=np.mean(df[DATA_COLUMN_NAME]),
    #     transition_offsets=np.mean(df[DATA_COLUMN_NAME]),
    #     # em_tol=1e-5
    # )
    #
    # # Model 4: ARIMA model without smoothing
    # model4 = KalmanFilter(
    #     n_dim_obs=1,
    #     n_dim_state=2,
    #     transition_matrices=np.array([[1, 1], [0, 1]]),
    #     observation_matrices=np.array([[1, 0]]),
    #     initial_state_mean=np.array([df[DATA_COLUMN_NAME].iloc[0], 0]),
    #     initial_state_covariance=np.diag([1, 1]),
    #     observation_covariance=1,
    #     transition_covariance=np.diag([0.1, 0.1]),
    #     observation_offsets=np.mean(df[DATA_COLUMN_NAME]),
    #     transition_offsets=np.mean(df[DATA_COLUMN_NAME]),
    #     em_vars=[]
    # )

    # Define a function to apply the Kalman filter
    def apply_kalman_filter(model, data):
        # Split the data into known and missing values
        known = data[~data['is_NA']]
        missing = data[data['is_NA']]

        # Apply the Kalman filter to the known values
        smoothed_state_means, _ = model.smooth(known[DATA_COLUMN_NAME].values)

        # Use the Kalman filter to impute missing values
        interpolated_values = model.filter(missing[DATA_COLUMN_NAME].values)[0][:, 0]

        # Combine the known and interpolated values
        all_values = np.concatenate([smoothed_state_means[:, 0], interpolated_values])

        # Calculate the space heating (SH) and heating water (DHW) use
        df['SH'] = np.where(df['is_NA'], all_values, df[DATA_COLUMN_NAME])
        df['DHW'] = np.where(df['is_NA'], df[DATA_COLUMN_NAME] - df['SH'], 0)

        return df

    # Create a new dataframe with the 'SH' and 'DHW' columns initialized to 0.
    df_new = df.copy()
    df_new["SH"] = 0.0
    df_new["DHW"] = 0.0

    # Apply the Kalman filter with each model
    if model_type == "smoothed structural":
        estimate_df = apply_kalman_filter(model1, df_new)
    elif model_type == "unsmoothed structural":
        estimate_df = apply_kalman_filter(model2, df_new)
    # elif model_type == "smoothed ARIMA":
    #     estimate_df = apply_kalman_filter(model3, df_new)
    # elif model_type == "unsmoothed ARIMA":
    #     estimate_df = apply_kalman_filter(model4, df_new)

    # Print the resulting dataframe
    return estimate_df


def SVR_estimate_method(df):
    df_copy = df.copy()

    weather = pd.read_csv(WEATHER_DATA_FILE_NAME, index_col=0, parse_dates=True)
    # Split the data into training and testing sets
    train = df_copy[df_copy['is_NA'] == False]
    test = df_copy[df_copy['is_NA'] == True]

    test['mean_temp'] = weather.loc[test.index.date, 'mean_temp'].values
    # print(test)
    # Train the SVR model using the training data
    X_train = weather.loc[train.index.date, 'mean_temp'].values.reshape(-1, 1)
    y_train = train[DATA_COLUMN_NAME].values

    svr = SVR(kernel='linear', C=1e3, epsilon=0.1)
    svr.fit(X_train, y_train)

    X_test = test['mean_temp'].values.reshape(-1, 1)
    y_pred = svr.predict(X_test)
    test['SH'] = y_pred

    test['DHW'] = test[DATA_COLUMN_NAME] - test['SH']

    df_new = pd.concat([train, test], axis=0)
    df_new_drop = df_new.drop(labels='mean_temp', axis=1)
    df_new_drop = df_new_drop.sort_index()

    return df_new_drop


def combine_kalman_SVR_estimate_method():
    ...


def separate_and_estimate_ETHOS_open_close(df):
    weekday_df_open, weekday_df_close, weekend_df_open, weekend_df_close = weekday_weekend_open_close_df(df)
    grouped_weekday_close = weekday_df_close.groupby(weekday_df_close.index.date)
    mean_grouped_weekday_close = grouped_weekday_close.mean()
    grouped_weekend_close = weekend_df_close.groupby(weekend_df_close.index.date)
    mean_grouped_weekend_close = grouped_weekend_close.mean()

    mean_grouped_close = pd.concat([mean_grouped_weekday_close, mean_grouped_weekend_close], axis=0).sort_index()

    week_df_open = pd.concat([weekday_df_open, weekend_df_open], axis=0).sort_index()

    # create a dictionary to map the values from df2 to dates
    base_dict = mean_grouped_close.squeeze().to_dict()

    # use the dictionary to map the values to the 'base' column in df1
    week_df_open['base'] = week_df_open.index.map(lambda x: base_dict.get(x.date()))

    week_df_open['other'] = week_df_open[DATA_COLUMN_NAME] - week_df_open["base"]

    df = pd.concat([week_df_open, weekday_df_close, weekend_df_close], axis=0).sort_index()
    # delete the replicated index
    df = df.groupby(df.index).last()
    # print(df)
    return df



def main_process(filename):
    # Load CSV file into a pandas DataFrame
    df = pd.read_csv(filename, parse_dates=[['Date', 'Time']], index_col=0)

    # Pre-processing with nan, negative and the datetime format
    df = data_preprocessing(df)
    # Add a new column with all False values
    df['is_NA'] = False
    # Select the column need to be used
    df = df[[DATA_COLUMN_NAME, 'is_NA']]
    print(len(df))

    # ============================================== Call method to separate the use of SH+DHW and SH=====================================
    separate_result_df = maximum_peaks_find(df, num_peaks=7, start_time='0500', end_time='2400')
    # separate_result_df = expected_profiles_find(df)
    # separate_result_df = outdoor_temperature_find(df)
    # separate_result_df = combine_find_method_1(df)

    # print(separate_result_df)
    print(len(separate_result_df))

    # ==============================================Call method to estimate the use of SH================================================
    # Interpolation method
    estimate_result_df = interpolation_estimate_method(separate_result_df, 'cubic spline')

    # Moving average window method
    # estimate_result_df = moving_average_estimate_method(separate_result_df, 'LMA')

    # Kalman filter, ARIMA not completed
    # estimate_result_df = kalman_filtering_estimate_method(separate_result_df, 'smoothed structural')

    # SVR
    # estimate_result_df = SVR_estimate_method(separate_result_df)

    print(estimate_result_df)

    # set negative values to zero
    estimate_result_df['DHW'] = estimate_result_df['DHW'].apply(lambda x: max(0, x))
    # set NaN to 0
    estimate_result_df = estimate_result_df.fillna(0, inplace=False)

    # rename column
    estimate_result_df = estimate_result_df.rename(columns={'SH': 'Base usage'})
    estimate_result_df = estimate_result_df.rename(columns={'DHW': 'Other usage'})

    # calculate standard derivation
    # print(df.std())

    # Plot the result df as line graph
    # estimate_result_df.plot(figsize=(30, 8))
    # plt.title('Result')
    # # Customize the plot
    # plt.xlabel('Date Time')
    # plt.ylabel('Data Value')
    # plt.legend(title='Consumption Type')
    #
    # result_method2_df.plot(figsize=(30, 8))
    # plt.title('Result of Method2')
    # # Customize the plot
    # plt.xlabel('Date Time')
    # plt.ylabel('Data Value')
    # plt.legend(title='Channel Names')

    # Display the plot
    # plt.show()

    estimate_result_df = estimate_result_df.iloc[:-1]

    # group the data by month
    grouped = estimate_result_df.groupby(estimate_result_df.index.month)

    # create subplots for each month's data
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))

    # iterate through each group and plot the data in the corresponding subplot
    for i, (month, data) in enumerate(grouped):
        ax = axes[i // 4, i % 4]
        data.plot(ax=ax, legend=False)
        ax.set_title('Month {}'.format(month))
        ax.set_xlabel('Date')
        ax.set_ylabel('Value(kWh)')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center')

    fig.tight_layout()
    plt.show()



