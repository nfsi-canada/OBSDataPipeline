"""
Parse *.btr files from Discovery OBS Locator, combine into a full ship track and downsample.

Author: K. Bosman
June 10, 2024
"""
import argparse
from datetime import datetime
from glob import glob
import os
import pandas as pd
import traceback

from nfsi_obs.io import parse_btr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse Discovery boat tracker files into downsampled CSV ship track.')
    parser.add_argument('--dir', dest="dir", help="Directory where *.btr files are stored.")
    parser.add_argument('--ds', dest="ds_rate", default='10min', help="Sample rate for downsampled GPS data.")

    try:
        args = parser.parse_args()
        run_start = datetime.now()

        # Base directories
        if args.dir:
            directory = os.path.abspath(os.path.expanduser(os.path.expandvars(args.dir)))
        else:
            raise SyntaxError('No input directory specified!')

        btr_files = glob(os.path.join(directory, '*.btr'))

        tracks = []
        for bf in btr_files:
            print('Reading {}...'.format(bf))
            tracks.append(parse_btr(bf))

        full_track = pd.concat(tracks)
        full_track.to_csv(os.path.join(directory, 'full_btr_parsed.csv'), index=False)

        print('Downsample to rate {}...'.format(args.ds_rate))
        down = full_track.resample(args.ds_rate)

        simplified_track = down.first()
        simplified_track.dropna(inplace=True)
        simplified_track.to_csv(os.path.join(directory, 'simplified_track_{}.csv'.format(args.ds_rate)), index=False)
        print('Simplified track saved to file.')

    except Exception as e:
        print(traceback.print_exc())
        parser.print_help()
        exit(1)
