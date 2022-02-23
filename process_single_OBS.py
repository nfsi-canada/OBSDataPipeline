import argparse
from glob import glob
import os
import pandas as pd
import re
import traceback
from datetime import datetime

import utilities as obsutil

# Ensure resource directory exists
resource_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'resource/OBSDataPipeline')
if not os.path.isdir(resource_dir):
    os.makedirs(resource_dir)


def process(data_dir, obs_log, output_dir=None):
    g_log.info("start")

    raw_files = glob(data_dir+'/*.mseed')
    g_log.info("Found {0} miniSEED files in data directory and sub-folders".format(len(raw_files)))
    
    # Metadata and pre-processing
    # TODO: Fix channel/station/network codes if necessary (N/E/Z vs 1/2/3)
    # TODO: Add station locations to metadata
    # TODO: Apply clock drift
    # TODO: Sensor orientation
    # TODO: Produce StationXML format metadata

    # Basic QC steps (seismic channels and hydrophone) -> if channel code == "CHx" or "HDF"
    # TODO: Decide if the same operations are appropriate for the hydrophone data or not
    # TODO: Calculate hourly PSDs
    # TODO: Average PSD value at 0.2 Hz (save out for comparison with other sensors in the same network)
    # TODO: Linearity of PSD curves

    # Analysis of auxiliary data
    # maybe smooth out state-of-health channels? or come up with some way to automatically QC them for anomalous sections
    # TODO: Plot battery draw-down and power consumption over full deployment
    # TODO: Plot internal state-of-health variables: pressure, temperature, humidity
    # TODO: Down-sample external pressure and temperature data (plot and save as netCDF)

    g_log.info("end")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process OBS data and perform basic QC')
    parser.add_argument('--data_dir', dest="data_dir", help="Directory where raw OBS data is stored.")
    parser.add_argument('--datalog', dest="datalog",
                        help="Log file from deployment/recovery. Must include station identifiers and clock drift "
                             "measurements. If not specified, assumed to be a file called 'log.xlsx' in the data "
                             "directory. Preferred format is XLSX (or similar spreadsheet) following NFSI template.")
    parser.add_argument('--logdelimiter', dest="log_delim", default=",",
                        help="If the OBS log file is delimited text (other than comma-delimited), use this to specify "
                             "the column delimiter.")
    parser.add_argument('--obsid', dest="obs_id", default="AQU-0000",
                        help="OBS identifier: station name or serial number")
    parser.add_argument('--network', "network_id", help="Network identifier assigned by FDSN for this project")
    parser.add_argument('--outdir', dest="outdir", default=None,
                        help="Output directory, if different from data directory")

    try:
        args = parser.parse_args()
        start_time = datetime.now()
        obs_id = args.obs_id

        g_log = obsutil.logger.get_general_logger(start_time, obs_id)
        g_log.info("\n\n=====================================================================")
        g_log.info("Starting job: {0}".format(str(args)))

        obs_identifier = args.obs_id
        id_type = 'unknown'
        if re.match(r'AQU-[0-9a-fA-F]{4}', obs_identifier):
            id_type = 'serial'
        elif re.match(r'D[aA][lL][_\-][0-9]{2,3}', obs_identifier):
            id_type = 'obs_name'

        if args.data_dir:
            data_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.data_dir)))
        else:
            data_dir = os.path.join(resource_dir, 'test_data')

        output_dir = None
        if args.outdir is not None:
            output_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.outdir)))
        if args.datalog:
            data_log_file = os.path.abspath(os.path.expanduser(os.path.expandvars(args.datalog)))
        else:
            data_log_file = os.path.join(data_dir, 'log.xlsx')

        obs_log_info = obsutil.parse_obs_log(data_log_file, args.log_delim)
        base_meta = obs_log_info['basic']
        # Find this OBS in the basic metadata table
        row = None
        if id_type == 'serial':
            row = base_meta.loc[base_meta['OBS ID'] == obs_identifier]
        elif id_type == 'obs_name':
            row = base_meta.loc[base_meta['OBS Name'] == obs_identifier]
        else:
            id_columns = ['Station', 'OBS Name', 'OBS ID']
            for col in id_columns:
                if obs_identifier in base_meta[col].values:
                    row = base_meta.loc[base_meta[col] == obs_identifier]
                    break

        if row is None or row.empty:
            raise(IndexError, 'OBS {0} not found in provided metadata.'.format(obs_identifier))
        if isinstance(row, pd.DataFrame):
            raise(IndexError, 'Multiple entries found for OBS {0} in provided metadata. Please use a unique identifier.'.format(obs_identifier))

        # TODO: Get clock drift info
        # TODO: Get station location info
        # TODO: Get network/station codes (if not correct in raw data)
        # Process data
        process(data_dir, row, output_dir)

        g_log.info("Processing complete!")
        obsutil.logger.close_logs()
    except Exception as e:
        print(traceback.print_exc())
        parser.print_help()
        exit(1)
