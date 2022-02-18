import argparse
import os
import traceback
from datetime import datetime

from utilities import logger

# Ensure resource directory exists
resource_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'resource/OBSDataPipeline')
if not os.path.isdir(resource_dir):
    os.makedirs(resource_dir)


def process():
    g_log.info("start")
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
    parser.add_argument('--obsid', dest="obs_id", default="AQU-0000", help="OBS identifier: station name or serial number")

    try:
        args = parser.parse_args()
        start_time = datetime.now()
        obs_id = args.obs_id

        g_log = logger.get_general_logger(start_time, obs_id)
        g_log.info("\n\n=====================================================================")
        g_log.info("Starting job: {0}".format(str(args)))

        # TODO: Get clock drift info
        # TODO: Get station location info
        # TODO: Get network/station codes (if not correct in raw data)
        # Process data
        process()

        g_log.info("Processing complete!")
        logger.close_logs()
    except Exception as e:
        print(traceback.print_exc())
        parser.print_help()
        exit(1)
