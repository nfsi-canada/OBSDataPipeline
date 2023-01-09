# OBSDataPipeline
Pre-processing and QC tasks for data collected from NFSI's OBS instruments

## Setup
### Environment
* This package uses a conda environment (Anaconda/Miniconda)
* Run `conda env create` to setup the environment or `conda env update` to update an existing environment after a package version change

### Configuration
* This package expects to be located at `[base_dir]/OBSDataPipeline`, with a folder called `resource` at the same level as `base_dir`.
* Copy `config/config.ini.stock` to `resource/OBSDataPipeline/config/` and name the copy `config.ini`. Alternatively, the script will do this automatically if `config.ini` does not exist.
* Edit `config.ini` as necessary for your particular setup 
