import argparse
from datetime import datetime
from glob import glob
import obspy
import os
import re
import timeit
import traceback
import warnings

from obspy.io.stationxml.core import validate_stationxml
from obstools.atacr import utils
import nfsi_obs as nf

# Ensure resource directory exists
resource_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'resource/OBSDataPipeline')
if not os.path.isdir(resource_dir):
    os.makedirs(resource_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare seismic and hydrophone data for tilt and compliance '
                                                 'correction processing with OBSTools package. Assumes data has been '
                                                 'saved as day-long miniSEED files.')
    parser.add_argument('--base_dir', dest='base_dir', help="Base directory where all files are stored (or will be "
                                                            "specified relative to).")
    parser.add_argument('--relative_paths', dest="relative_paths", action="store_true",
                        help="If true, all other path arguments are specified relative to the base directory.")
    parser.add_argument('--data_dir', dest="data_dir", help="Directory where OBS data is stored.")
    parser.add_argument('--obsid', dest="obs_id",
                        help="OBS identifier: station name or serial number")
    parser.add_argument('--start', dest="startdate", help="Start date of time period to be analyzed, as YYYYMMDD")
    parser.add_argument('--end', dest="enddate", help="End date of time period to be analyzed, as YYYYMMDD")
    parser.add_argument('--network', dest="network_id",
                        help="Network identifier assigned by FDSN for this project. Default 'XX' for test data.")
    parser.add_argument('--outdir', dest="outdir",
                        help="Output directory, if different from data directory")
    parser.add_argument('--channelmap', dest="channel_map",
                        help="File mapping as-recorded channel codes to their correct values.")
    parser.add_argument('--datalog', dest="datalog",
                        help="Log file from deployment/recovery. Must include station identifiers and clock drift "
                             "measurements. If not specified, assumed to be a file called 'log.xlsx' in the data "
                             "directory. Preferred format is XLSX (or similar spreadsheet) following NFSI template.")
    parser.add_argument('--logdelimiter', dest="log_delim", default=',',
                        help="If the OBS log file is delimited text (other than comma-delimited), use this to specify "
                             "the column delimiter.")
    parser.add_argument('--metadata', dest="metadata_file",
                        help="Path to metadata file (dataless SEED or StationXML). If not specified, will search "
                             "data_dir for a suitable file.")
    parser.add_argument('--O', dest="overwrite", action="store_true", help="Overwrite existing output data files.")

    # Copied from OBStools atacr_download_data.py
    FreqGroup = parser.add_argument_group(title='Frequency Settings', description="Miscellaneous frequency settings.")
    FreqGroup.add_argument('--sampling_rate', type=float, dest="new_sampling_rate", default=5.,
                           help="Specify new sampling rate (float, in Hz). [Default 5.]")
    FreqGroup.add_argument('--units', type=str, dest="units", default="DISP",
                           help="Choose the output seismogram units. Options are: DISP, VEL, ACC. [Default DISP]")
    FreqGroup.add_argument('--pre-filt', type=str, dest="pre_filt", default=None,
                           help="Specify four comma-separated corner frequencies for deconvolution pre-filter. "
                                "[Default 0.001,0.005,45.,50.]")

    try:
        start_time = datetime.now()
        t0 = timeit.default_timer()
        args = parser.parse_args()

        if args.units not in ['DISP', 'VEL', 'ACC']:
            raise(Exception("Error: invalid --units argument. Choose from 'DISP', 'VEL' or 'ACC'."))
        if args.pre_filt is None:
            args.pre_filt = [0.001, 0.005, 45., 50.]
        else:
            args.pre_filt = [float(val) for val in args.pre_filt.split(',')]
            args.pre_filt = sorted(args.pre_filt)
            if len(args.pre_filt) != 4:
                raise(Exception('Error: --pre-filt should contain 4 comma-separated floats'))

        if args.base_dir:
            base_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.base_dir)))
        else:
            base_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(os.path.join(resource_dir, 'test_data'))))

        if args.obs_id:
            obsid = args.obs_id
        else:
            obsid = 'AQU-0260'
        id_type = 'unknown'
        if re.match(r'AQU-[0-9a-fA-F]{4}', obsid):
            id_type = 'serial'
        elif re.match(r'D[aA][lL][_\-][0-9]{2,3}', obsid):
            id_type = 'obs_name'

        tstart, tend = None, None
        if args.startdate:
            tstart = datetime.strptime(args.startdate, '%Y%m%d')
        if args.enddate:
            tend = datetime.strptime(args.enddate, '%Y%m%d')

        if args.data_dir:
            data_path = args.data_dir
        else:
            data_path = os.path.join(base_dir, 'AQU-0260')
        if args.relative_paths:
            data_dir = os.path.join(base_dir, data_path)
        else:
            data_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(data_path)))
        data_dir = os.path.normpath(data_dir)

        output_dir, out_path = None, None
        if args.outdir:
            out_path = args.outdir
            if args.relative_paths:
                output_dir = os.path.join(base_dir, out_path)
            else:
                output_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(out_path)))
            output_dir = os.path.normpath(output_dir)
        else:
            output_dir = base_dir
        dataout = os.path.join(output_dir, 'DATA', obsid)
        if not os.path.exists(dataout):
            os.makedirs(dataout)

        # Deployment summary info (for station locations)
        if args.datalog:
            datalog = args.datalog
            if args.relative_paths:
                data_log_file = os.path.join(base_dir, datalog)
            else:
                data_log_file = os.path.abspath(os.path.expanduser(os.path.expandvars(datalog)))
            data_log_file = os.path.normpath(data_log_file)
        else:
            data_log_file = os.path.join(base_dir, 'log.xlsx')

        print('Reading deployment summary from {0}...'.format(data_log_file))
        obs_log_info = nf.io.parse_obs_log(data_log_file, args.log_delim)
        # Find this OBS in the basic, deployment, and recovery metadata tables
        base_meta, dep, rec = None, None, None
        if id_type == 'serial':
            base_meta = obs_log_info['basic'].loc[obs_log_info['basic']['OBS ID'] == obsid]
            dep = obs_log_info['deployment'].loc[obs_log_info['deployment']['OBS ID'] == obsid]
            rec = obs_log_info['recovery'].loc[obs_log_info['recovery']['OBS ID'] == obsid]
        elif id_type == 'obs_name':
            base_meta = obs_log_info['basic'].loc[obs_log_info['basic']['OBS Name'] == obsid]
            dep = obs_log_info['deployment'].loc[obs_log_info['deployment']['OBS Name'] == obsid]
            rec = obs_log_info['recovery'].loc[obs_log_info['recovery']['OBS Name'] == obsid]
        else:
            id_columns = ['Station', 'OBS Name', 'OBS ID']
            for col in id_columns:
                if obsid in obs_log_info['basic'][col].values:
                    base_meta = obs_log_info['basic'].loc[obs_log_info['basic'][col] == obsid]
                    dep = obs_log_info['deployment'].loc[obs_log_info['deployment'][col] == obsid]
                    rec = obs_log_info['recovery'].loc[obs_log_info['recovery'][col] == obsid]
                    break

        if base_meta is None or base_meta.empty:
            raise IndexError('OBS {0} not found in provided metadata.'.format(obsid))
        if base_meta.shape[0] > 1:
            raise IndexError('Multiple entries found for OBS {0} in provided metadata. Please use a unique identifier.'.format(obsid))

        inv = obs_log_info['inventory'].select(station=base_meta['Station'].values[0])
        sta = inv.networks[0].stations[0]

        # Read station metadata file
        station_info = None
        if args.metadata_file:
            meta_file = args.metadata_file
            if args.relative_paths:
                metadata_file = os.path.join(base_dir, meta_file)
            else:
                metadata_file = os.path.abspath(os.path.expanduser(os.path.expandvars(meta_file)))
            print("Reading metadata from file {0}".format(metadata_file))
            filetype = os.path.splitext(metadata_file)[-1]
            if filetype in ['.dataless', '.metadata']:
                station_info = nf.metadata.read_dataless(metadata_file)
            elif filetype == '.xml':
                # read as StationXML format
                station_info = obspy.read_inventory(metadata_file)
            else:
                print("Unrecognized file format. Unable to read metadata.")
        else:
            # Search data_dir for suitable metadata file
            seed_files = glob(os.path.join(data_dir, '**/*.dataless'), recursive=True)
            seed_files.extend(glob(os.path.join(data_dir, '**/*.metadata'), recursive=True))
            xml_files = glob(os.path.join(data_dir, '**/*.xml'), recursive=True)
            if len(seed_files) > 0:
                if len(seed_files) > 1:
                    print("Multiple dataless SEED volumes found in data directory, using {0}.".format(seed_files[0]))
                else:
                    print("Found dataless SEED volume in data directory: {0}".format(seed_files[0]))
                # take first dataless SEED file
                station_info = nf.metadata.read_dataless(seed_files[0])
            elif len(xml_files) > 0:
                for xf in xml_files:
                    is_sxml = validate_stationxml(xf)[0]
                    if is_sxml and (station_info is None):
                        print("Found StationXML file in data directory: {0}".format(xf))
                        station_info = obspy.read_inventory(xf)
            else:
                warnings.warn("No metadata file provided, and none found in data directory.")

        channel_map = None
        if args.channel_map:
            ch_map = args.channel_map
            if args.relative_paths:
                channel_map = nf.io.read_channel_map(os.path.join(base_dir, ch_map))
            else:
                channel_map = nf.io.read_channel_map(ch_map)
        else:
            print("No channel map provided. Channel IDs will be processed as they appear in the raw data files.")

        if args.network_id:
            network_id = args.network_id
        else:
            network_id = 'XX'

        datafiles = glob(os.path.join(data_dir, '**', '*.mseed'), recursive=True)

        for df in datafiles:
            print('Reading {}'.format(os.path.basename(df)))
            data = obspy.read(df)
            # Populate metadata from other files as necessary
            data = nf.metadata.update_metadata(data, network_id, station_info=station_info, channel_map=channel_map)

            no_writing = True
            if args.overwrite:
                no_writing = False
            else:
                for tr in data:
                    filename = str(tr.stats.starttime.year).zfill(4) + '.' + str(tr.stats.starttime.julday).zfill(3) + '.' + tr.id + '.SAC'
                    if not os.path.isfile(os.path.join(dataout, filename)):
                        no_writing = False

            if no_writing:
                print('Output file(s) already exist, skipping...')
                continue

            # Replicate data filtering/downsampling from OBStools atacr_download_data.py
            data.detrend('demean')
            data.detrend('linear')
            data.filter('lowpass', freq=0.5*args.new_sampling_rate, corners=2, zerophase=True)
            data.resample(args.new_sampling_rate)

            data.remove_response(pre_filt=args.pre_filt, output=args.units)

            for tr in data:
                tr = utils.update_stats(tr, sta.latitude, sta.longitude, sta.elevation, tr.stats.channel)
                tr.write(os.path.join(dataout, str(tr.stats.starttime.year).zfill(4) + '.' + str(tr.stats.starttime.julday).zfill(3) + '.' + tr.id + '.SAC'), format='SAC')

        end_time = datetime.now()
        run_time = timeit.default_timer() - t0
        if run_time < 60:
            print("Total run time: {0} seconds".format(run_time))
        elif run_time < 3600:
            print("Total run time: {0} seconds ({1} minutes)".format(run_time, run_time/60))
        elif run_time < 3600*24:
            print("Total run time: {0} seconds ({1} hours)".format(run_time, run_time/3600))
        else:
            print("Total run time: {0} seconds ({1} days)".format(run_time, run_time/3600/24))

    except Exception as e:
        print(traceback.print_exc())
        parser.print_help()
        exit(1)
