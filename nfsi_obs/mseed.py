"""
Operations performed on miniSEED files directly (bypassing obspy for lower-level control)
"""
from bitstring import ConstBitStream
from datetime import datetime, timedelta
import io
import numpy as np
import re


class MiniSEED:
    """A class representing a miniSEED file"""

    def __init__(self, mseed_file_path):
        """
        Opens the file at mseed_file_path and splits it into data records

        :param mseed_file_path: Path to miniSEED data file
        """
        self.mseed_file = mseed_file_path
        self.metadata = {}
        self.data_records = []

    def read_headers(self):
        """
        Read header information for all data records in miniSEED file

        :return: Number of data records found
        """
        mseed = ConstBitStream(io.open(self.mseed_file))

        # Read headers of all data records from miniSEED file buffer
        while mseed.pos < len(mseed):
            record = {'header': {}, 'data': []}

            # read data record fixed header
            record['header']['sequence'] = mseed.read('bytes:6').decode('ascii')
            record['header']['quality'] = mseed.read('bytes:1').decode('ascii')
            mseed.read('bytes:1')   # empty reserved byte
            record['header']['station'] = mseed.read('bytes:5').decode('ascii')
            record['header']['location'] = mseed.read('bytes:2').decode('ascii')
            record['header']['channel'] = mseed.read('bytes:3').decode('ascii')
            record['header']['network'] = mseed.read('bytes:2').decode('ascii')
            year = mseed.read('uint:16')
            daynum = mseed.read('uint:16')
            hour = mseed.read('uint:8')
            minute = mseed.read('uint:8')
            second = mseed.read('uint:8')
            mseed.read('uint:8')    # unused in BTIME format
            millisec_tenth = mseed.read('uint:16')
            record['header']['start_time'] = datetime(year, 1, 1, hour, minute, second, millisec_tenth*100) + timedelta(daynum-1)
            record['header']['num_samples'] = mseed.read('uint:16')
            record['header']['sample_rate_factor'] = mseed.read('uint:16')
            record['header']['sample_rate_multiplier'] = mseed.read('uint:16')
            record['header']['activity_flags'] = mseed.read('bin:8')
            record['header']['io_clock_flags'] = mseed.read('bin:8')
            record['header']['data_quality_flags'] = mseed.read('bin:8')
            record['header']['num_following_blockettes'] = mseed.read('uint:8')
            record['header']['time_correction'] = mseed.read('uint:32')
            record['header']['data_start'] = mseed.read('uint:16')  # length of entire header section
            record['header']['first_blockette'] = mseed.read('uint:16') # fixed header length

            record['header']['subblocks'] = []
            try:
                # read other blockettes
                for i in range(record['header']['num_following_blockettes']):
                    block = {}
                    block['type'] = mseed.read('uint:16')
                    if block['type'] == 1000:
                        # Data only SEED blockette
                        block['next_blockette'] = mseed.read('uint:16')
                        block['encoding'] = mseed.read('uint:8')
                        block['word_order'] = mseed.read('uint:8')
                        block['record_length'] = mseed.read('uint:8')
                        mseed.read('bytes:1')   # reserved byte
                        record['header']['encoding'] = block['encoding']
                        record['header']['word_order'] = block['word_order']
                        record['header']['record_length'] = block['record_length']
                    elif block['type'] == 1001:
                        # data extension blockette
                        block['next_blockette'] = mseed.read('uint:16')
                        block['timing_quality'] = mseed.read('uint:8')
                        block['microseconds'] = mseed.read('uint:8')
                        mseed.read('bytes:1')   # reserved byte
                        block['frame_count'] = mseed.read('uint:8')
                        record['header']['timing_quality'] = block['timing_quality']
                        record['header']['microseconds'] = block['microseconds']
                        record['header']['frame_count'] = block['frame_count']
                    else:
                        raise(Exception, "Unimplemented block type {0}".format(block['type']))

                    record['header']['subblocks'].append(block)

                    # check that record length indicators are consistent
                    if pow(2, record['header']['record_length']) != (64 * (record['header']['frame_count'] + 1)):
                        raise(ArithmeticError, "Data record length 2^{0} and frame count 64*{1} do not agree!".format(record['header']['record_length'], record['header']['frame_count']))
            except Exception:
                pass

            self.data_records.append(record)
            # Move cursor to start of next data record
            mseed.pos = pow(2, record['header']['record_length']) * 8

        # Gather overall metadata for the entire file
        first_record = self.data_records[0]
        for key in ['network', 'station', 'location', 'channel', 'word_order', 'encoding', 'record_length']:
            self.metadata[key] = first_record['header'][key]
        self.metadata['channel_id'] = '{0}.{1}.{2}.{3}'.format(first_record['header']['network'], first_record['header']['station'], first_record['header']['location'], first_record['header']['channel'])
        record_start = [x['header']['start_time'] for x in self.data_records]
        i_earliest = np.argmin(record_start)
        i_latest = np.argmax(record_start)
        self.metadata['start_time'] = self.data_records[i_earliest]['header']['start_time'] + timedelta(microseconds=(self.data_records[i_earliest]['header']['microseconds']))
        latest_sample_rate_info = [self.data_records[i_latest]['header']['sample_rate_factor'], self.data_records[i_latest]['header']['sample_rate_multiplier']]
        if latest_sample_rate_info[0] > 0:
            if latest_sample_rate_info[1] > 0:
                latest_sample_rate = latest_sample_rate_info[0] * latest_sample_rate_info[1]
            else:
                latest_sample_rate = -1 * latest_sample_rate_info[0] / latest_sample_rate_info[1]
        else:
            if latest_sample_rate_info[1] > 0:
                latest_sample_rate = latest_sample_rate_info[1] * latest_sample_rate_info[0]
            else:
                latest_sample_rate = 1 / (latest_sample_rate_info[0] * latest_sample_rate_info[1])
        self.metadata['end_time'] = self.data_records[i_latest]['header']['start_time'] \
                                    + timedelta(microseconds=(self.data_records[i_latest]['header']['microseconds'])) \
                                    + timedelta(seconds=((self.data_records[i_latest]['header']['num_samples'] - 1) / latest_sample_rate))

        return len(self.data_records)

    def clock_drift_correction(self, total_drift):
        """
        Correct data record start times for clock drift. Assumes linear drift between start and end of recording period.

        :param total_drift: Total clock drift in milliseconds
        :return:
        """
