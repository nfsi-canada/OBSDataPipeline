"""
Add missing info to existing StationXML file(s) from Aquarius, and reduce to only channels actually recorded.

Author: K. Bosman
June 7, 2024
"""
import argparse
import warnings
from datetime import datetime, timedelta
from glob import glob
import json
import numpy as np
import obspy
import os
import pandas as pd
import re
import traceback

from obspy.io.stationxml.core import validate_stationxml
from obspy.core.inventory import Inventory, Network, Station, Channel, Operator, Person, Equipment, PhoneNumber

import nfsi_obs as nf
from utilities import logger


def filter_inventory(inv, channel_list):
    """
    Filter obspy.Inventory object to only channel IDs included in `channel_list`.
    """
    for ch in inv.get_contents()['channels']:
        if ch not in channel_list:
            codes = ch.split('.')
            inv = inv.remove(network=codes[0], station=codes[1], location=codes[2], channel=codes[3])

    return inv.copy()


def map_and_filter_xml(sxml_file, channel_list=None, channel_map=None, is_dataless=False):
    """
    Filter StationXML file to only channels included in list of channels, applying ID mapping specified in
    `channel_map`. If `channel_list` is not specified, filter to only channels which appear in `channel_map`. If no
    extra arguments provided, no filtering will be performed.

    :param sxml_file: Path to StationXML file
    :param channel_list: List of channel IDs
    :param channel_map: pandas.DataFrame mapping existing channel IDs to corrected IDs
    :param is_dataless: bool, set to True if `sxml_file` is a dataless SEED file
    :return: obspy.Inventory
    """
    # Read input metadata file to obspy.Inventory object
    if is_dataless:
        input_inv = nf.metadata.read_dataless(sxml_file)
    else:
        is_sxml = validate_stationxml(sxml_file)
        if not is_sxml:
            raise TypeError('Input file {} is not a valid StationXML file.'.format(sxml_file))

        input_inv = obspy.read_inventory(sxml_file)

    if channel_list is None and channel_map is None:
        # No filtering to be done, return inventory as-is
        return input_inv

    input_channels = input_inv.get_contents()['channels']

    # Pull relevant channel mapping information
    mapped_channel_ids = None
    if channel_map is not None:
        mapped_id_list = []
        for ch in input_channels:
            try:
                ch_info = channel_map.loc[ch]
            except KeyError as e:
                # channel ID not in channel map
                continue

            if ch_info.empty:   # channel not in map, ignore
                continue

            mapped_id_list.append(ch_info)

        if len(mapped_id_list) > 0:
            mapped_channel_ids = pd.concat(mapped_id_list, axis=1).transpose()

    # Filter output channel list to only those specified
    if channel_list is not None:
        if channel_map is None:
            # Simple filter, no ID mapping to be done
            return filter_inventory(input_inv, channel_list)
    elif mapped_channel_ids is not None:
        channel_list = mapped_channel_ids['Correct channel ID'].values
    else:
        g_log.warning('No channels in XML match list to filter.')
        return Inventory()

    # Channel ID mapping
    obj_refs = {}
    for ch in input_channels:
        try:
            ch_info = channel_map.loc[ch]
        except KeyError as e:
            # channel ID not in channel map
            continue
        if ch_info.empty:  # channel not in map, leave as-is
            continue

        codes = ch.split('.')
        orig_channel = input_inv.select(network=codes[0], station=codes[1], location=codes[2], channel=codes[3])

        new_ch = ch_info['Correct channel ID']
        new_codes = new_ch.split('.')
        if (new_codes[0] not in [n.code for n in input_inv.networks]) or ('Net_{}'.format(new_codes[0]) not in obj_refs):
            # Network not present in inventory, copy from original coded Network (no stations/channels)
            old_network = input_inv.select(network=codes[0])
            new_net = old_network.networks[0].copy()
            new_net.code = new_codes[0]
            new_net.stations = []
            input_inv.networks.append(new_net)
            obj_refs['Net_{}'.format(new_codes[0])] = new_net
        else:
            new_net = obj_refs['Net_{}'.format(new_codes[0])]

        if new_codes[1] not in [s.code for s in new_net.stations]:
            # Station not present in correct network, copy from original coded Station (no channels)
            old_station = input_inv.select(network=codes[0], station=codes[1])
            new_sta = old_station.networks[0].stations[0].copy()
            new_sta.code = new_codes[1]
            new_sta.channels = []
            new_net.stations.append(new_sta)
            obj_refs['Sta_{}'.format(new_codes[1])] = new_sta
        else:
            new_sta = obj_refs['Sta_{}'.format(new_codes[1])]

        new_channel = orig_channel.networks[0].stations[0].channels[0].copy()
        new_channel.code = new_codes[3]
        new_channel.location_code = new_codes[2]
        if not pd.isnull(ch_info['Description']):
            new_channel.description = ch_info['Description']
        new_sta.channels.append(new_channel)

    # Filter re-mapped inventory
    return filter_inventory(input_inv, channel_list)


def update_station_xml(inv, obs_log=None, extra_info=None, nfsi_fields=False, survey_method='Triangulation'):
    """
    Add/update info in obspy.Inventory to fit StationXML standard. Station/channel coordinates are taken from `obs_log`.
    Various other metadata fields are in the `extra_info` dictionary.
    """
    if nfsi_fields:
        # General information, NFSI-specific
        inv.source = 'NFSI'
        inv.module = 'OBSDataPipeline 0.4.0'
        inv.module_uri = 'https://github.com/nfsi-canada/OBSDataPipeline'

    # Station/channel coordinates
    for n in inv.networks:
        if nfsi_fields:
            n.operators = [Operator('NFSI',
                                  contacts=[
                                      Person(agencies=['NFSI'],
                                             emails=['nfsi@nfsi.ca'],
                                             phones=[PhoneNumber(902, '494-6130', country_code=1)]),
                                  ],
                                  website='https://nfsi.ca')]

        net_info = None
        if 'network_{}'.format(n.code) in extra_info:
            net_info = extra_info['network_{}'.format(n.code)]
            if 'source_id' in net_info:
                n.source_id = net_info['source_id']
            if 'restricted_status' in net_info:
                n.restricted_status = net_info['restricted_status']
            if 'description' in net_info:
                n.description = net_info['description']
            if 'identifiers' in net_info:
                n.identifiers = ['{0}:{1}'.format(idf['type'], idf['value']) for idf in net_info['identifiers']]

        for s in n.stations:
            base_meta = obs_log['basic'].loc[obs_log['basic']['Station'] == s.code]
            lat = base_meta['Deployed Latitude'].values[0]
            lon = base_meta['Deployed Longitude'].values[0]
            elev = -base_meta['Water Depth (m)'].values[0]
            start = obspy.UTCDateTime(pd.to_datetime(base_meta['Date/Time on Seafloor (UTC)'].values[0]))
            end = obspy.UTCDateTime(pd.to_datetime(base_meta['Date/Time Released (UTC)'].values[0]))
            try:
                survey_method = base_meta['Survey Calculation Method'].values[0]
            except (KeyError, IndexError):
                # Column not present, use default
                pass

            s.latitude = lat
            s.longitude = lon
            s.elevation = elev
            s.latitude.__setattr__('measurement_method', survey_method)
            s.longitude.__setattr__('measurement_method', survey_method)
            s.elevation.__setattr__('measurement_method', survey_method)
            s.water_level = 0
            s.start_date = start
            s.end_date = end

            sta_info = None
            if net_info is not None:
                if 'station_{}'.format(s.code) in net_info:
                    sta_info = net_info['station_{}'.format(s.code)]
                    if 'water_level' in sta_info:
                        s.water_level = sta_info['water_level']

            for c in s.channels:
                c.latitude = lat
                c.longitude = lon
                c.elevation = elev
                c.latitude.__setattr__('measurement_method', survey_method)
                c.longitude.__setattr__('measurement_method', survey_method)
                c.elevation.__setattr__('measurement_method', survey_method)
                c.start_date = start
                c.end_date = end

                if nfsi_fields:
                    if c.code == 'MDO':
                        # External pressure sensor, have serial numbers for Keller sensors
                        c.sensor = Equipment(description='Piezoresistive absolute pressure transducer',
                                             manufacturer='KELLER', model='PA-10L', serial_number='FILL_FROM_DB')
                    elif c.code == 'HDH':
                        # Broadband hydrophone
                        c.sensor = Equipment(description='Ultra low frequency broadband hydrophone',
                                             manufacturer='High Tech, Inc.', model='HTI-04-PCA/ULF',
                                             serial_number='FILL_FROM_DB')
                    else:
                        # Same sensor/equipment info as Station (Aquarius)
                        c.sensor = None

                if sta_info is not None:
                    if 'channel_{}'.format(c.code) in sta_info:
                        ch_info = sta_info['channel_{}'.format(c.code)]
                        if 'sensor' in ch_info:
                            if 'serial_number' in ch_info['sensor']:
                                c.sensor.serial_number = ch_info['sensor']['serial_number']

    return inv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Complete StationXML files using partial files created by Aquarius '
                                                 'OBS and supplemental metadata files.')
    parser.add_argument('--input_dir', dest="in_dir",
                        help="Directory where input StationXML files are stored, and/or base directory for relative "
                             "paths.")
    parser.add_argument('--output_dir', dest="out_dir", help="Directory where output files are to be stored.")
    parser.add_argument('--log_dir', dest="log_dir", help="Directory to store log files.")
    parser.add_argument('--relative_paths', dest="relative_paths", action="store_true",
                        help="Specify all file paths relative to in_dir.")
    parser.add_argument('--xml', dest="aqu_xml",
                        help="Path to input StationXML file (channel IDs not corrected by channel_map) if only editing "
                             "a single file. Will override directory of files if specified.")
    parser.add_argument('--datalog', dest="datalog",
                        help="Log file from deployment/recovery. Must include station identifiers and locations. "
                             "Preferred format is XLSX (or similar spreadsheet) following NFSI template.")
    parser.add_argument('--legacylogcols', dest="obslog_column_names_legacy", action="store_true",
                        help="Use legacy column names for OBS deployment log file.")
    parser.add_argument('--channelmap', dest="channel_map",
                        help="File mapping as-recorded channel codes to their correct values.")
    parser.add_argument('--out_channels', dest="out_channels", default=None,
                        help="List of channel IDs to include in output XML file (after any required correction to "
                             "codes). Either comma-separated string or file (comma-separated or one channel per line). "
                             "If not specified, channels included in channel_map will be output. If no channel_map is "
                             "specified, all channels will be output.")
    parser.add_argument('--other_meta', dest="other_metadata",
                        help="JSON file with various metadata to be added to StationXML files.")
    parser.add_argument('--dataless', action='store_true', dest="is_dataless",
                        help="Flag to set if input files are dataless SEED rather than StationXML (legacy option).")
    parser.add_argument('--survey', dest="survey_method", default='Triangulation',
                        help="Survey method used for determining seafloor locations. Default 'Triangulation'.")

    try:
        args = parser.parse_args()
        run_start = datetime.now()

        # Base directories
        input_dir = None
        if args.in_dir:
            input_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.in_dir)))

        if args.relative_paths:
            if input_dir is None:
                raise RuntimeError('Missing command-line argument: Cannot use relative paths if in_dir not specified.')

        if args.aqu_xml:
            if args.relative_paths:
                xml_files = [os.path.abspath(os.path.expanduser(os.path.expandvars(os.path.join(input_dir, args.aqu_xml))))]
            else:
                xml_files = [os.path.abspath(os.path.expanduser(os.path.expandvars(args.aqu_xml)))]
        elif input_dir is not None:
            if args.is_dataless:
                xml_files = glob(os.path.join(input_dir, '**', '*.dataless'), recursive=True)
            else:
                xml_files = glob(os.path.join(input_dir, '**', '*.xml'), recursive=True)
        else:
            raise SyntaxError('No input file or directory specified!')

        out_dir = None
        if args.out_dir:
            if args.relative_paths:
                out_dir = os.path.join(input_dir, args.out_dir)
            else:
                out_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.out_dir)))

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        # Logging setup
        log_dir = None
        if args.log_dir:
            if args.relative_paths:
                log_dir = os.path.join(input_dir, args.log_dir)
            else:
                log_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.log_dir)))

        if log_dir is not None:
            g_log = logger.get_general_logger(run_start, 'XML', logs_dir=log_dir)
        else:
            g_log = logger.get_general_logger(run_start, 'XML')
        g_log.info("\n\n=====================================================================")
        g_log.info("Starting job: {0}".format(str(args)))

        if args.is_dataless:
            g_log.info("Found {} dataless SEED files to edit".format(len(xml_files)))
        else:
            g_log.info("Found {} XML files to edit".format(len(xml_files)))

        # Channel list (if specified separately)
        channels = None
        if args.out_channels:
            if args.relative_paths:
                ch_path = os.path.join(input_dir, args.out_channels)
            else:
                ch_path = args.out_channels

            if os.path.isfile(ch_path):
                channel_file = os.path.abspath(os.path.expanduser(os.path.expandvars(ch_path)))
                cf = open(channel_file)
                ch_list = cf.readlines()
                if len(ch_list) > 1:
                    channels = [c for c in ch_list]
                else:
                    channels = ch_list[0].split(',')
            else:
                channels = args.out_channels.split(',')

        # Metadata files
        obs_log_info = None
        if args.datalog:
            if args.relative_paths:
                data_log_file = os.path.normpath(os.path.join(input_dir, args.datalog))
            else:
                data_log_file = os.path.normpath(os.path.abspath(os.path.expanduser(os.path.expandvars(args.datalog))))
            g_log.info('Reading project metadata from {0}...'.format(data_log_file))
            obs_log_info = nf.io.parse_obs_log(data_log_file, names_in_file=not args.obslog_column_names_legacy)

        channel_map = None
        if args.channel_map:
            if args.relative_paths:
                ch_map = os.path.normpath(os.path.join(input_dir, args.channel_map))
            else:
                ch_map = os.path.abspath(os.path.expanduser(os.path.expandvars(args.channel_map)))
            channel_map = nf.io.read_channel_map(ch_map)

        extra_meta = None
        if args.other_metadata:
            if args.relative_paths:
                meta_json = os.path.normpath(os.path.join(input_dir, args.other_metadata))
            else:
                meta_json = os.path.abspath(os.path.expanduser(os.path.expandvars(args.other_metadata)))
            extra_meta = json.load(open(meta_json))

        # Output directory fallbacks -> input_dir -> where XML found
        if out_dir is None:
            if input_dir is not None:
                out_dir = input_dir

        for xf in xml_files:
            print(xf)
            # Fix channel identifiers and filter to channels of interest
            good_channels = map_and_filter_xml(xf, channels, channel_map, args.is_dataless)
            num_chan = int(np.sum([len(s.channels) for n in good_channels.networks for s in n.stations]))
            print('Filtered channels: {}'.format(num_chan))
            if num_chan < 1:
                continue

            complete_metadata = update_station_xml(good_channels, obs_log_info, extra_meta, nfsi_fields=True, survey_method=args.survey_method)

            # Save output XML file
            if len(complete_metadata.networks) > 1:
                out_file = os.path.basename(xf)
            else:
                try:
                    net = complete_metadata.networks[0].code
                except IndexError as e:
                    continue

                if len(complete_metadata.networks[0].stations) > 1:
                    out_file = '{}.xml'.format(net)
                else:
                    try:
                        sta = complete_metadata.networks[0].stations[0].code
                    except IndexError as e:
                        continue

                    out_file = '{}_{}.xml'.format(net, sta)

            if out_dir is None:
                out_path = os.path.join(os.path.dirname(xf), 'filtered_{}'.format(out_file))
            else:
                out_path = os.path.join(out_dir, out_file)
            complete_metadata.write(out_path, format='STATIONXML', validate=True)

        g_log.info("Processing complete!")
        # TODO: Combine individual stations into full-network StationXML file
        # TODO: Full-network StationXML should have start/end dates for network equal to earliest start and latest end for any station
        logger.close_logs()
    except Exception as e:
        print(traceback.print_exc())
        parser.print_help()
        exit(1)
