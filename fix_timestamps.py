"""
Attempt to fix miniSEED data record start times being incorrect.
"""
from datetime import datetime, timedelta
from glob import glob
import obspy
from obspy.io.mseed.util import get_start_and_end_time
import os
from struct import unpack, pack

base_dir = 'L:/Data/Ischia test deployment'
subfolders = ['AQU-8063-time-corrected']

outdir = os.path.join(base_dir, 'Corrected')
if not os.path.exists(outdir):
    os.makedirs(outdir)

for sf in subfolders:
    print('Checking folder {}...'.format(sf))
    data_files = glob(os.path.join(base_dir, sf, '**', '*.mseed'), recursive=True)

    output_dir = os.path.join(outdir, sf)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for df in data_files:
        print('Reading {}...'.format(df))
        try:
            temp = obspy.read(df, header_byteorder='>')
            times = get_start_and_end_time(df)
        except Exception as e:
            print(str(e))

            bits = open(df, 'rb')
            newbits = b''
            newfile = os.path.join(output_dir, os.path.basename(df))
            if os.path.isfile(newfile):
                print('Corrected file already exists. Skipping...')
                continue    # skip existing corrected files

            #nf = open(newfile, 'ab')
            bo = '>'
            while True:
                record = bits.read(4096)
                fmt = '%sHHBBBxH' % bo
                if record:
                    timeinfo = unpack(fmt, record[20:30])
                    print('\r' + str(timeinfo), end="")

                    if not (1970 <= timeinfo[0] <= 2599):
                        bo = '<'
                        timeinfo = unpack('<HHBBBxH', record[20:30])
                        if not (1970 <= timeinfo[0] <= 2599):
                            print('\nUnknown byte order, skipping this record.')
                            continue
                        print('\nLittle-endian byte order')
                    if not (1 <= timeinfo[1] <= 366):
                        #print('Julian day incorrect, fixing...')
                        startdate = datetime(timeinfo[0], 1, 1) + timedelta(days=timeinfo[1] - 1)
                        year = startdate.year
                        julday = int(startdate.strftime('%j'))
                        newtime = (year, julday, *timeinfo[2:])
                    else:
                        newtime = timeinfo

                    newrecord = record[:20] + pack(fmt, *newtime) + record[30:]
                    #nf.write(newrecord)
                    newbits += newrecord
                else:
                    break

            nf = open(newfile, 'wb')
            nf.write(newbits)

            print('\nWrote corrected data to file {}, length {}'.format(newfile, len(newbits)))
            nf.close()
            bits.close()


