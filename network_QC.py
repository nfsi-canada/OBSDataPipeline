"""
Check noise levels across an entire deployed network. Basic QC must have been applied to all individual OBS in the
network prior to running this script (process_single_OBS.py).
"""
import argparse

# TODO: Read in noise level output from process_single_OBS.py for all sensors
# TODO: Calculate average power level for each channel-hour
# TODO: DBSCAN clustering of channel-hour power levels
# TODO: Label channel-hours as good or anomalous
# TODO: Plot results and output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network-level QC')
    parser.add_argument('--networkid', dest='network_id', required=True, help='Network identifier')
