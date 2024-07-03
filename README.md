# OBSDataPipeline

This package is developed and maintained by the NFSI for quality control assessment and pre-processing operations performed on data acquired by NFSI's Aquarius ocean-bottom seismometers (OBS). The code is subject to ongoing maintenance and development, so instructions for use may change from time to time. Reasonable efforts will be made to maintain backwards compatibility.

## Setup

For generating PDF reports, an installation of pandoc and texlive is required. All other package requirements are included in the conda environment definition.

### Environment

* This package uses a conda environment (Anaconda/Miniconda), running Python 3
* Run `conda env create` to setup the environment or `conda env update` to update an existing environment after a package version change

#### Dependencies

- Python 3
- numpy (1.21)
- matplotlib
- scipy
- obspy (1.3.1 or greater)
- pandas
- openpyxl
- bitstring
- pypandoc
- jinja2
- scikit-learn
- seaborn
- [stdb](https://github.com/schaefferaj/StDb)
- [obstools](https://github.com/nfsi-canada/OBStools)
- pynmeagps
- [ioos_qc](https://github.com/kaelbos/ioos_qc) (modified to allow sample rate greater than 1Hz)

### Configuration

* By default, this package expects to be located at `[base_dir]/OBSDataPipeline`, with a folder called `resource` at the same level as `base_dir`. If this is not the case, the code cannot be run without inputs (default test case) and `log_dir` must be specified in the `common` section of the default config file.
* Copy `configs/config.ini.stock` to either `resource/OBSDataPipeline/configs/` or the existing `configs` directory and name the copy `config.ini`. Alternatively, the script will do this automatically the first time it runs if `config.ini` does not exist.
* Edit `config.ini` as necessary for your particular setup.

## Field Data QC (`OBS_QC.py`)

This script is intended to be run on raw data downloaded from the OBS. It may be run at any time after downloading the data, but is currently only designed to be run on the Aquarius-specific data package (data in miniSEED format, metadata as dataless SEED or StationXML, data for each auxiliary channel spans a maximum of 3 miniSEED files).

General settings are included in a `config.ini` file (default example provided), which can typically be specified at a project level. Station-specific settings are normally few enough to use command line arguments without being too cumbersome. Most settings may be specified either as command line arguments or parameters in the INI file, except of course the path to the INI file itself if not using the default one.

The script `QC_many.py` is designed to run this QC for several instruments in sequence, with a JSON file specifying all command line parameters. This is especially useful for test datasets, where several instruments will be "recovered" at the same time, or for batch updating field data after a recovery cruise.

### Available Settings

#### Command line and config.ini:

|         Name         |          Default Value          | Description                                                                                                                                             |
|:--------------------:|:-------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------|
|       `config`       |                                 | Path to `config.ini` file                                                                                                                               |
|      `base_dir`      |                                 | Base directory if using relative paths                                                                                                                  |
|   `relative_paths`   |              False              | Flag to specify all other paths relative to `base_dir`                                                                                                  |
|      `data_dir`      |                                 | Directory where OBS data is stored (script will find ALL miniSEED files in this directory)                                                              |
|      `datalog`       |                                 | Deployment summary spreadsheet, including deployment and recovery information. Preferred format is XLSX following NFSI template.                        |
|    `logdelimiter`    |                                 | Optional delimiter if `datalog` file is delimited text.                                                                                                 |
|    `logcolnames`     |              False              | Maintained for backwards compatibility. If true, use column names from `datalog` file, rather than hard-coded (legacy) names.                           |
|       `obsid`        |                                 | Unique identifier for OBS (normally station name or OBS serial number)                                                                                  |
|       `start`        |                                 | Start date of deployment to be analyzed (if multiple deployments of same instrument present in log file). Normally only required for test datasets.     |
|      `network`       |               XX                | Network code assigned for the project                                                                                                                   |
|       `outdir`       |           `data_dir`            | Output directory for QC results                                                                                                                         |
|     `channelmap`     |                                 | Spreadsheet or delimited text file mapping correct SEED codes to existing channel identifiers in raw data. Optional                                     |
|      `metadata`      |                                 | Optional path to metadata file (dataless SEED or StationXML). If not specified, code will search `data_dir` for a suitable file                         |
|     `extra_meta`     |                                 | Optional JSON file with extra description, QC information and per-channel plot settings                                                                 |
|  `detrend_seismic`   |              False              | Detrend seismic data prior to analysis (RMS linear fit)                                                                                                 |
|    `skip_backup`     |              False              | Do not create a backup of raw data files                                                                                                                |
| `use_existing_plots` |              False              | Do not re-create plots which already exist in output directory                                                                                          |
|    `projectname`     |                                 | Project name. If not specified, code looks in `extra_meta` JSON file instead.                                                                           |
|      `colormap`      |             viridis             | Matplotlib colormap to use for spectrograms                                                                                                             |
|       `debug`        |              False              | Set logging level to debug (see package `logging`)                                                                                                      |

#### Config.ini only:

All other parameters belong in section `dataset`.

|         Name         |  Section  |          Default Value          | Description                                                                                                                                             |
|:--------------------:|:---------:|:-------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------|
|      `log_dir`       | `common`  | ~/resource/OBSDataPipeline/logs | Directory where run-time logs are to be saved                                                                                                           |
|    `pdftex_path`     | `common`  |                                 | Path to PDFLaTex install directory                                                                                                                      |
|   `window_length`    | `seismic` |              3600               | Window length to be used for PSD calculations, in seconds (default 1 hour)                                                                              |
|  `overlap_percent`   | `seismic` |               50                | Percent overlap for PSD windows (default 50%)                                                                                                           |
| `spectrogram_window` | `seismic` |               60                | Window length to be used for spectrogram, in seconds (default 1 minute). Set equal to PSD window length (not yet properly implemented to be different). |

### Intended Folder Structure

- base_dir
  - station_1_dir
  - station_2_dir
  - ...
  - station_n_dir
  - Project_Deployment_Summary.xlsx
  - project_info.json
  - project_channel_map.xlsx (if required)

## Convert Aquarius Data to SDS Archive (`miniseed_recut.py`)

Raw data from the Aquarius OBS is in miniSEED file format, with one channel per file and maximum file size of 128 MB. This means that the amount of data in a particular file varies with sampling rate and compression efficiency.

The standard used by SeisComP (SDS archive) has data in miniSEED format, with each file including data for a single channel over a 24-hour period (UTC day). These files are organized in a standard folder structure, and have standardized filenames.

The `SDS_many.py` script allows this operation to be run for several stations in sequence with a JSON file providing command line inputs.

- Base folder
  - Year
    - Network
      - Station
        - Channel
          - miniSEED data files

Filename: [Net].[Sta].[Loc].[Chan].[Year].[JulianDay].mseed
- Example XX.L102..CH3.2023.352.mseed

## Correct and Complete StationXML (`edit_stationxml.py`)

The Aquarius OBS produce StationXML files which include all response information for channels available on the instrument. Values for location may be set during programming of the instrument, but these are generally not known accurately prior to deployment. This script updates all relevant values from the project/station metadata, and corrects SEED codes as necessary.
