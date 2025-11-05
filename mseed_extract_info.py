import argparse
from datetime import datetime
from glob import glob
import numpy as np
import os
import pandas as pd
import timeit
import warnings

from obspy.io.mseed.util import get_start_and_end_time, get_record_information


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract basic information from all miniSEED files in a directory.")
    parser.add_argument('base_dir', type=str,
                        help="Path to folder to be analyzed.")

    run_start = timeit.default_timer()
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.base_dir)))

    # Find data files
    raw_files = glob(os.path.join(base_dir, '**/*.mseed'), recursive=True)

    # Ignore any files calculated by previous QC script runs
    try:
        raw_files.remove(os.path.join(base_dir, 'calculated_current.mseed'))
    except ValueError:
        pass
    dspk = glob(os.path.join(base_dir, '**/*_despiked.mseed'), recursive=True)
    for df in dspk:
        try:
            raw_files.remove(df)
        except ValueError:
            pass

    print("Found {0} miniSEED file(s) in data directory and sub-folders".format(len(raw_files)))

    # Extract start/end timestamps
    file_info = []
    for rf in raw_files:
        print(os.path.basename(rf))
        times = np.array(get_start_and_end_time(rf))
        if np.any(times < datetime(2021, 9, 1)):
            warnings.warn('Some data timestamps prior to 2021-09-01 (invalid).')
        info = get_record_information(rf)
        file_info.append({
            'file_name': os.path.basename(rf),
            'channel_id': '{}.{}.{}.{}'.format(info['network'], info['station'], info['location'], info['channel']),
            'start_time': times[0],
            'end_time': times[1],
        })

    file_info_df = pd.DataFrame(file_info)
    file_info_df.to_csv(os.path.join(base_dir, 'miniSEED_file_information.csv'), index=False)

    run_end = timeit.default_timer()

    print('Total runtime: {0:.3f} seconds ({1:.3f} minutes)'.format(run_end - run_start, (run_end - run_start) / 60))
