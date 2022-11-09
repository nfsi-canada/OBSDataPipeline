import os
import pandas as pd
import re
import warnings

from utilities import check_nan


def convert_dataless_to_stationxml(dataless_file, base_meta, output_dir, channel_map=None):
    """
    Convert a dataless SEED file to StationXML format.

    :param dataless_file: Path to dataless SEED file
    :param base_meta: Basic metadata from OBS field log (single row from result of io.parse_obs_log())
    :type base_meta: pd.Series
    :param output_dir: Path to output directory
    :param channel_map: Table mapping recorded channel IDs to corrected identifiers, optional
    :type channel_map: pd.DataFrame or None

    :return: Path to output StationXMl file
    """
    from obspy.io.xseed import Parser
    from obspy.io.xseed.core import _parse_to_inventory_object

    metadata = Parser(dataless_file)
    meta_inv = _parse_to_inventory_object(metadata)

    # TODO: Add station locations to metadata (from base_meta)
    # TODO: Write Inventory object to StationXML format (file `xml_out`)

    xml_out = os.path.join(output_dir, 'whatever.xml')
    return xml_out


def read_dataless(dataless_file):
    """
    Read a dataless SEED file and return an obspy.core.inventory.Inventory object

    :param dataless_file: full path to dataless SEED file
    :return:
    """
    from obspy.io.xseed import Parser
    from obspy.io.xseed.core import _parse_to_inventory_object

    metadata = Parser(dataless_file)
    # Get network/station/channel info as an Inventory object
    meta_inv = _parse_to_inventory_object(metadata)

    return meta_inv


def update_metadata(data, network_id, log, station_info=None, channel_map=None, project_meta=None):
    """
    Update metadata for each trace in obspy.Stream object from other sources.

    :param data: input time series data as obspy.core.stream.Stream object
    :param network_id: 2-character FDSN network code
    :param log: Logging object used by calling script
    :param station_info
    :param channel_map
    :param project_meta

    :return: updated Stream object
    """
    for tr in data:
        # Get response info from metadata
        if station_info is not None:
            try:
                tr.attach_response(station_info)
            except ValueError:
                warnings.warn("No matching response information found")

            # Get orientations of seismic channels
            if re.match(r'[A-Z]H[1-3ABCENRTUVWZ]', tr.meta.channel):
                orient = station_info.get_orientation(tr.id)
                for key in ['azimuth', 'dip']:
                    tr.stats[key] = orient[key]

        # Fix channel/station/network codes if necessary (N/E/Z vs 1/2/3)
        if channel_map is not None:
            ch_info = channel_map.loc[tr.id]
            for code in ['Network', 'Station', 'Location', 'Channel', 'Description']:
                if ch_info[code] is not None and ~check_nan(ch_info[code]):
                    tr.meta[code.lower()] = ch_info[code]
        if tr.meta.network != network_id:
            raise AssertionError('Channel {0} is not in network {1}'.format(tr.id, network_id))

        # Get channel info from project metadata JSON
        if project_meta is not None:
            try:
                channel_info = list(filter(lambda ch: ch['channel_id'] == tr.meta.channel, project_meta['channels']))[0]
                if 'description' in channel_info:
                    tr.meta.description = channel_info['description']
            except (KeyError, IndexError):
                log.warning("No matching information found in project metadata for channel {0}".format(tr.id))

    return data


def get_channel_type(ch_code):
    # Assign to appropriate group of channels
    channel_type = 'health'
    if (re.match(r'[A-Z]H[1-3ABCENRTUVWZ]', ch_code)) or (re.match(r'[A-Z]D[HF]', ch_code)):
        # seismic data and hydrophone
        channel_type = 'seismic'
    elif ch_code in ['LKO', 'MDO', 'MDU']:
        # oceanographic data (external P/T, include APG if present)
        # TODO: Would like this to be more general, but internal temperature is also labeled with "KO" source/subsource code by default
        channel_type = 'ocean'
    elif ch_code in ['LE3', 'ME4']:
        # battery voltage and power consumption
        channel_type = 'power'

    return channel_type
