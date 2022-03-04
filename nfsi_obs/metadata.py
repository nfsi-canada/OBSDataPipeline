import os
import pandas as pd


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
