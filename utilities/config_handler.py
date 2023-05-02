import io
import os
import configparser
import shutil

def get_config(config_path=None):
    # The config file contains various settings used by this program
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stock_config_path = os.path.join(base_dir, 'configs/config.ini.stock')
    if config_path is None:
        resource_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'resource/OBSDataPipeline')
        config_path = os.path.join(resource_dir, 'configs/config.ini')

    # Check if there is a config file, and if not, copy the stock config file
    if not os.path.isfile(config_path):
        if not os.path.isfile(stock_config_path):
            # There's no config file or stock config file
            raise IOError('There is no configuration file or stock configuration file - unable to process data')
        else:
            os.makedirs(os.path.dirname(config_path))
            # Copy the stock config file to the expected config file location
            shutil.copyfile(stock_config_path, config_path)

    # Load info from the config file
    config = configparser.ConfigParser()
    config.read(config_path)
    config['dataset']['config_path'] = config_path
    return config


def copy_config(old_config):
    config_string = io.StringIO()
    old_config.write(config_string)
    config_string.seek(0)

    new_config = configparser.ConfigParser()
    new_config.read(config_string)
    return new_config
