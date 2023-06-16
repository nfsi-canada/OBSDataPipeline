"""
Take in arbitrary length miniSEED files for a given channel, and output data as day-long miniSEED files.

Author: K. Bosman
June 6, 2023
"""
from datetime import datetime, timedelta
from glob import glob
import obspy
import os
import traceback

# TODO: Make script callable with arguments (easier to reuse)

channels_of_interest = ['SeisE', 'SeisN', 'SeisZ', 'SeisX']
base_dir = 'L:/Data/Ischia test deployment'
#subfolders = ['AQU-4261', 'AQU-8263', 'AQU-B063', 'AQU-8063-fixed']
subfolders = ['AQU-8063-fixed']

outdir = os.path.join(base_dir, 'Day-long mseed')
if not os.path.exists(outdir):
    os.makedirs(outdir)

for sf in subfolders:
    print('Checking folder {}...'.format(sf))
    for ci in channels_of_interest:
        print('Looking for channel {}...'.format(ci))
        data_files = glob(os.path.join(base_dir, sf, '**', '*'+ci+'*.mseed'), recursive=True)

        full_data = obspy.Stream()
        for df in data_files:
            print('Reading {}...'.format(df))
            try:
                temp = obspy.read(df, header_byteorder='>')
                for tr in temp:
                    full_data.append(tr)
            except Exception as e:
                #print(traceback.print_exc())
                print(str(e))
                continue

        full_data.merge()
        print(full_data)
        full_data.print_gaps()

        output_dir = os.path.join(outdir, sf)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for tr in full_data:
            start = tr.stats.starttime.datetime
            end = tr.stats.endtime.datetime + timedelta(days=1)
            startday = start.date()
            endday = end.date()

            cut = obspy.UTCDateTime(startday)
            while cut < endday:
                temp = tr.slice(cut, cut + 24 * 60 * 60, nearest_sample=False)
                stt = temp.split()  # deal with traces with gaps
                print(stt)

                outfile = os.path.join(output_dir, '{}.{}.{}.mseed'.format(cut.year, cut.julday, tr.id))
                stt.write(outfile, format="MSEED")

                cut += 24 * 60 * 60
