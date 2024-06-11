# OBSDataPipeline
Pre-processing and QC tasks for data collected from NFSI's OBS instruments

## Setup
### Environment
* This package uses a conda environment (Anaconda/Miniconda)
* Run `conda env create` to setup the environment or `conda env update` to update an existing environment after a package version change

### Configuration
* By default, this package expects to be located at `[base_dir]/OBSDataPipeline`, with a folder called `resource` at the same level as `base_dir`. If this is not the case, the code cannot be run without inputs (default test case) and `log_dir` must be specified in the `common` section of the default config file.
* Copy `configs/config.ini.stock` to either `resource/OBSDataPipeline/configs/` or the existing `configs` directory and name the copy `config.ini`. Alternatively, the script will do this automatically the first time it runs if `config.ini` does not exist.
* Edit `config.ini` as necessary for your particular setup.
