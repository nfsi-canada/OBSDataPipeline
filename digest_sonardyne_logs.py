import argparse
from glob import glob
import os
import pandas as pd
import re
import timeit
import warnings

from nfsi_obs.io import parse_usbl_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Digest log files from Ranger 2 USBL system.")
    parser.add_argument('folder', type=str, help="Path to folder containing log files.")

    full_start = timeit.default_timer()
    args = parser.parse_args()

    # Get file list in folder
    directory = os.path.abspath(os.path.expanduser(os.path.expandvars(args.folder)))
    if not os.path.isdir(directory):
        raise IOError('Input folder does not exist: {}'.format(os.path.normpath(directory)))
    all_files = glob(os.path.join(directory, '*.txt'))

    time_info = []

    for lf in all_files:
        istart = timeit.default_timer()

        filename = os.path.basename(lf)
        if re.match(r'[0-9]{8}_[0-9]{6}_[A-Za-z]+_[0-9]+_[A-Za-z0-9]_Log#[0-9]+[-\w]*\.txt', filename):
            # valid log file name
            log_info = parse_usbl_log(lf)
            log_info.to_csv(os.path.join(directory, 'parsed_'+os.path.splitext(filename)+'.csv'))
        else:
            warnings.warn('Invalid log file: {}'.format(filename))

        iend = timeit.default_timer()
        time_info.append([filename, iend - istart])

    full_end = timeit.default_timer()

    print('Runtime by file:')
    for ti in time_info:
        print('{0}: {1:.3f} seconds ({2:.3f} minutes)'.format(ti[0], ti[1], ti[1] / 60))

    print('Total runtime: {0:.3f} seconds ({1:.3f} minutes)'.format(full_end - full_start, (full_end - full_start) / 60))
