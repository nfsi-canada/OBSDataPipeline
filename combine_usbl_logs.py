import argparse
from datetime import datetime
from glob import glob
import numpy as np
import os
import pandas as pd
import timeit
import warnings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Combine parsed log files (as CSV) for Ranger 2 USBL system.")
    parser.add_argument('folder', type=str, help="Path to folder containing CSV log files.")
    parser.add_argument('starttime', type=str, help="Start date and time, as YYYYMMDD-HHMMSS")
    parser.add_argument('endtime', type=str, help="End date and time, as YYYYMMDD-HHMMSS")

    full_start = timeit.default_timer()
    args = parser.parse_args()

    start_time = datetime.strptime(args.starttime, '%Y%m%d-%H%M%S')
    end_time = datetime.strptime(args.endtime, '%Y%m%d-%H%M%S')

    # Get file list in folder
    directory = os.path.abspath(os.path.expanduser(os.path.expandvars(args.folder)))
    if not os.path.isdir(directory):
        raise IOError('Input folder does not exist: {}'.format(os.path.normpath(directory)))
    all_files = glob(os.path.join(directory, '*.csv'))

    time_info = []
    sonar_data = []
    other_data = []

    # Read data from all files and filter to time window of interest
    for lf in all_files:
        sonardyne = False
        istart = timeit.default_timer()

        filename = os.path.basename(lf)

        print(filename)
        try:
            log_data = pd.read_csv(lf, index_col=0, parse_dates=['datetime'])
        except ValueError:
            try:
                log_data = pd.read_csv(lf, index_col=0, parse_dates=['clock_time'])
            except ValueError:
                log_data = pd.read_csv(lf, index_col=0)

        if 'clock_time' in log_data:
            tk = 'clock_time'
            sonardyne = True
        elif 'datetime' in log_data:
            tk = 'datetime'
        else:
            warnings.warn('No recognized datetime key in table, skipping.')
            continue

        windowed = log_data[log_data[tk] > start_time]
        windowed = windowed[windowed[tk] < end_time]
        print('{} rows within time window'.format(windowed.shape[0]))
        if windowed.shape[0] < 1:   # No date in time window
            continue

        if sonardyne:
            filtered = windowed[windowed['sensor_type'] != 'GPGGA']
            if filtered.shape[0] < 1:
                continue

            print('{} rows of good data added to database'.format(filtered.shape[0]))
            filtered.set_index('clock_time', inplace=True)
            sonar_data.append(filtered)
        else:
            print('{} rows of good data added to database'.format(windowed.shape[0]))
            if 'message_type' in windowed:
                for name, group in windowed.groupby('message_type'):
                    print(name)
                    if name in ['GPGGA', 'GPZDA']:
                        bytype = group.set_index('datetime')
                        other_data.append(bytype)
            else:
                windowed.set_index('datetime', inplace=True)
                windowed.drop_duplicates(inplace=True)
                other_data.append(windowed)

        iend = timeit.default_timer()
        time_info.append([filename, iend - istart])

    # Combine log files and interpolate times as necessary
    all_sonar = pd.concat(sonar_data)
    all_sonar.sort_index(inplace=True)
    sonar_keys = []
    for k in all_sonar.keys():
        if all_sonar[k].count() > 0:
            sonar_keys.append(k)
    good_sonar = all_sonar[sonar_keys]
    # interpolate TOD column for transceiver messages
    time_only = good_sonar[good_sonar['sensor_category'] == 'TOD']
    time_strs = []
    for idx, x in time_only.iterrows():
        time_bits = [int(x['year']), int(x['month']), int(x['day']), x['time']]
        time_strs.append('{:04d}-{:02d}-{:02d} {}'.format(*time_bits))
    time_data = np.array([np.datetime64(x) for x in time_strs])
    interp_time = np.interp(good_sonar.index, time_only.index, time_data.astype(float))
    good_sonar['interpolated_UTC'] = pd.Series(pd.to_datetime(interp_time * 1e3), index=good_sonar.index)

    good_sonar.to_csv(os.path.join(directory, 'sonardyne_logs_combined.csv'))

    all_other = pd.concat(other_data)
    all_other.sort_index(inplace=True)
    other_keys = []
    for k in all_other.keys():
        if all_other[k].count() > 0:
            other_keys.append(k)
    good_other = all_other[other_keys]
    good_other.to_csv(os.path.join(directory, 'other_logs_combined.csv'))

    # Add GPS position info to sonardyne logs
    gps_position = good_other[good_other['message_type'] == 'GPGGA']
    good_sonar.set_index('interpolated_UTC', drop=False)
    lat_interp = np.interp(good_sonar.index, gps_position.index, gps_position['lat'].values)
    lon_interp = np.interp(good_sonar.index, gps_position.index, gps_position['lon'].values)
    good_sonar['latitude_i'] = pd.Series(lat_interp, index=good_sonar.index)
    good_sonar['longitude_i'] = pd.Series(lon_interp, index=good_sonar.index)

    good_sonar.to_csv(os.path.join(directory, 'sonardyne_logs_interpolatedGPS.csv'))

    full_end = timeit.default_timer()

    print('Runtime by file:')
    for ti in time_info:
        print('{0}: {1:.3f} seconds ({2:.3f} minutes)'.format(ti[0], ti[1], ti[1] / 60))

    print('Total runtime: {0:.3f} seconds ({1:.3f} minutes)'.format(full_end - full_start, (full_end - full_start) / 60))
