import os
import configparser
import shutil

def get_config():
    # The config file contains various settings used by this program
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resource_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'resource/OBSDataPipeline')
    config_path = os.path.join(resource_dir, 'configs/config.ini')
    stock_config_path = os.path.join(base_dir, 'configs/config.ini.stock')

    # Check if there is a config file, and if not, copy the stock config file
    if not os.path.isfile(config_path):
        if not os.path.isfile(stock_config_path):
            # There's no config file or stock config file
            raise IOError('There is no configuration file or stock configuration file - unable to process data')
        else:
            # Copy the stock config file to the expected config file location
            shutil.copyfile(stock_config_path, config_path)

    # Load info from the config file
    config = configparser.ConfigParser()
    config.read(config_path)
    return config
