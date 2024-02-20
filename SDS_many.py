"""
Run SDS archive setup script (miniseed_recut.py) for several instruments sequentially.

Author: K. Bosman
February 15, 2024
"""
import argparse
import json
import os
import subprocess
import timeit
import warnings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SDS archive script (miniseed_recut.py) for several instruments sequentially.")
    parser.add_argument('bulk_info', type=str,
                        help="JSON file containing command-line options for QC script for all instruments as a "
                             "dictionary.")

    full_start = timeit.default_timer()
    args = parser.parse_args()


    # Read input JSON file
    bulk_file = os.path.abspath(os.path.expanduser(os.path.expandvars(args.bulk_info)))
    if not os.path.isfile(bulk_file):
        raise IOError('Input JSON file does not exist: {}'.format(os.path.normpath(bulk_file)))
    bf = open(bulk_file)
    bulk = json.load(bf)

    if 'instruments' in bulk:
        instruments = bulk.pop('instruments')
    else:
        instruments = []
        warnings.warn('No instruments specified in input JSON file.')

    flags = None
    if 'flags' in bulk:
        flags = bulk.pop('flags')

    time_info = []

    for inst in instruments:
        istart = timeit.default_timer()

        args_list = [
            'python',
            'miniseed_recut.py',
        ]
        for bkey in bulk:
            args_list.append('--{0}={1}'.format(bkey, bulk[bkey]))
        for ikey in inst:
            args_list.append('--{0}={1}'.format(ikey, inst[ikey]))
        if flags is not None:
            for f in flags:
                args_list.append('--{0}'.format(f))

        subprocess.run(args_list)

        iend = timeit.default_timer()
        time_info.append([inst['subfolders'], iend - istart])

    full_end = timeit.default_timer()

    print('Runtime by instrument:')
    for ti in time_info:
        print('{0}: {1:.3f} seconds ({2:.3f} minutes)'.format(ti[0], ti[1], ti[1] / 60))

    print('Total runtime: {0:.3f} seconds ({1:.3f} minutes)'.format(full_end - full_start, (full_end - full_start) / 60))
