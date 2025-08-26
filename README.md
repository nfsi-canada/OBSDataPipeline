# OBSDataPipeline

This package is developed and maintained by the NFSI for quality control assessment and pre-processing operations performed on data acquired by NFSI's Aquarius ocean-bottom seismometers (OBS). The code is subject to ongoing maintenance and development, so instructions for use may change from time to time. Reasonable efforts will be made to maintain backwards compatibility.

## Setup

For generating PDF reports, an installation of pandoc and texlive is required. All other package requirements are included in the conda environment definition.

### Environment

* This package uses a conda environment (Anaconda/Miniconda), running Python 3
  * Python 3.7 to 3.10, inclusive, are compatible with the full dependency list at this time
* Run `conda env create` to setup the environment or `conda env update` to update an existing environment after a package version change

#### Dependencies

- Python 3 (3.7 to 3.10)
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
- psutil
- [stdb](https://github.com/schaefferaj/StDb)
- [obstools](https://github.com/nfsi-canada/OBStools)
- pynmeagps
- [ioos_qc](https://github.com/kaelbos/ioos_qc) (modified to allow sample rate greater than 1Hz)

### Configuration

* By default, this package expects to be located at `[base_dir]/OBSDataPipeline`, with a folder called `resource` at the same level as `base_dir`. If this is not the case the package will still function, but the data QC script (OBS_QC.py) cannot be run without inputs (default test case) and `log_dir` must be specified in the `common` section of the default config file.
* To use the data QC script (OBS_QC.py):
  * Copy `configs/config.ini.stock` to either `resource/OBSDataPipeline/configs/` or the existing `configs` directory in the repository and name the copy `config.ini`. Alternatively, the script will do this automatically the first time it runs if `config.ini` does not exist.
  * Edit `config.ini` as necessary for your particular setup.
  * Some basic settings fall back to this default configuration file if not specified in an individual project's `config.ini` file
* Configuration is not required for other functionality, including editing of StationXML information and data pre-processing for distribution.

## Field Data QC (`OBS_QC.py`)

This script is intended to be run on raw data downloaded from the Aquarius OBS. It may be run at any time after downloading the data, but is currently only designed to be run on the Aquarius-specific data package as downloaded from the OBS (data in miniSEED format, metadata as dataless SEED or StationXML, data for each auxiliary channel spans a maximum of 3 miniSEED files).

General settings are included in a `config.ini` file, which can typically be specified at a project level. Station-specific settings are normally few enough to use command line arguments without being too cumbersome. Most settings may be specified either as command line arguments or parameters in the INI file, except of course the path to the INI file itself if not using the default one. In the case of settings specified by multiple input methods, the command line arguments will take precedence.

The script `QC_many.py` is designed to run this QC for several instruments in sequence, with a JSON file specifying all command line parameters for each instrument. This is especially useful for test datasets, where several instruments will be "recovered" at the same time, or for batch updating field data after a recovery cruise.

### Available Settings

#### Command line only

|         Name         |          Default Value          | Description                                                                                                                      |
|:--------------------:|:-------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------|
|       `config`       |                                 | Path to `config.ini` file. Must be absolute path unless `base_dir` and `relative_paths` are also given as command line arguments |

#### Command line and config.ini:

Command line arguments override parameters also present in the configuration file.

|         Name         | Default Value | Description                                                                                                                                                      |
|:--------------------:|:-------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
|      `base_dir`      |               | Absolute path to base directory if using relative paths                                                                                                          |
|   `relative_paths`   |     False     | Flag to specify all other paths relative to `base_dir`                                                                                                           |
|      `data_dir`      |               | Directory where OBS data is stored (script will find ALL miniSEED files in this directory). Station-specific.                                                    |
|      `datalog`       |               | Deployment summary spreadsheet, including deployment and recovery information. Preferred format is XLSX following NFSI template.                                 |
|    `logdelimiter`    |       ,       | Optional delimiter if `datalog` file is delimited text (default comma-delimited)                                                                                 |
|    `logcolnames`     |     False     | Maintained for backwards compatibility. If true, use column names from `datalog` file, rather than hard-coded (legacy) names. Will be deprecated in future.      |
|       `obsid`        |               | Unique identifier for OBS (normally station name or OBS serial number)                                                                                           |
|       `start`        |               | Start date of deployment to be analyzed (if multiple deployments of same instrument present in log file), as YYYYMMDD. Normally only required for test datasets. |
|      `network`       |      XX       | FDSN-style network code assigned for the project                                                                                                                 |
|       `outdir`       |  `data_dir`   | Output directory for QC results                                                                                                                                  |
|     `channelmap`     |               | Spreadsheet or delimited text file mapping correct SEED codes to existing channel identifiers in raw data. Optional                                              |
|      `metadata`      |               | Optional path to metadata file (dataless SEED or StationXML). If not specified, code will search `data_dir` for a suitable file.                                 |
|     `extra_meta`     |               | Optional JSON file with extra description, QC information and per-channel plot settings                                                                          |
|  `detrend_seismic`   |     False     | Detrend seismic data prior to analysis (RMS linear fit)                                                                                                          |
| `use_existing_plots` |     False     | Do not re-create plots which already exist in output directory (not fully implemented)                                                                           |
|    `projectname`     |               | Project name. If not specified, code looks in `extra_meta` JSON file instead.                                                                                    |
|      `colormap`      |    viridis    | Matplotlib colormap to use for spectrograms                                                                                                                      |
|   `ignore_seismic`   |     False     | Do not analyze data channels for seismometer or hydrophone                                                                                                       |
|  `limited_seismic`   |     False     | Analysis of seismometer and hydrophone data limited to only data extent, readability, and gaps                                                                   |
|       `debug`        |     False     | Set logging level to debug (see package `logging`)                                                                                                               |

#### Config.ini only:

All CLI-optional parameters listed above belong in section `dataset`.

|         Name         |  Section  |          Default Value          | Description                                                                                                                                                              |
|:--------------------:|:---------:|:-------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|      `log_dir`       | `common`  | ~/resource/OBSDataPipeline/logs | Directory where run-time logs are to be saved. Will recognize `:base` and `:data` as `base_dir` and `data_dir`, respectively. Otherwise, specify full path.              |
|    `pdftex_path`     | `common`  |                                 | Path to PDFLaTex application executable                                                                                                                                  |
|   `window_length`    | `seismic` |              3600               | Window length to be used for PSD calculations, in seconds (default 1 hour)                                                                                               |
|  `overlap_percent`   | `seismic` |               50                | Percent overlap for PSD windows (default 50%)                                                                                                                            |
| `spectrogram_window` | `seismic` |               60                | NOT IMPLEMENTED. Window length to be used for spectrogram, in seconds (default 1 minute). Set equal to PSD window length (not yet properly implemented to be different). |

### Intended Folder Structure

The code does not require a specific folder structure outside of `data_dir` itself. Below is a recommended structure for ease of organization and clarity.

- base_dir
  - station_1_dir
  - station_2_dir
  - ...
  - station_n_dir
  - Project_Deployment_Summary.xlsx
  - project_info.json
  - project_channel_map.xlsx (if required)

Additionally, a directory named `QC reports` is often created under the base directory to keep QC results separate from the raw data. Each station's QC output is then saved in a sub-directory named as the station ID or OBS identifier.

## Convert Aquarius Data to SDS Archive (`miniseed_recut.py`)

Raw data as downloaded directly from the Aquarius OBS is in miniSEED file format, with one channel per file and maximum file size of 128 MB, using the STEIM2 data compression algorithm. This means that the time span covered by any particular file varies with sampling rate and compression efficiency.

The standard used by SeisComP (SDS archive) and more familiar to seismology researchers has data in miniSEED format, with each file including data for a single channel over a 24-hour period (UTC day). These files are organized in a standard folder structure, and have standardized filenames. See the [SeisComP documentation](https://docs.gempa.de/seiscomp3/current/apps/slarchive.html) for more information.

The `miniseed_recut.py` script can be used to convert raw Aquarius data to the SDS structure. The `SDS_many.py` script allows this operation to be run for several stations in sequence with a JSON file providing command line inputs.

- Base folder
  - Year
    - Network
      - Station
        - Channel (optional ".D" suffix)
          - miniSEED data files

Filename ("data" channel): [Net].[Sta].[Loc].[Chan].D.[Year].[JulianDay].mseed
- Example 2S.L102..CH3.D.2023.352.mseed
- Auxiliary channels omit the ".D" from the file name and channel folder name

Some channel identifiers used by default on the Aquarius OBS do not follow the SEED convention, and require correction using the same channel map file used by the data QC script. This pre-processing script also applies a linear clock drift correction based on the final clock offset measurement collected at instrument recovery, if available.

### CLI Parameters

|        Name        | Default Value | Description                                                                                                                                                                      |
|:------------------:|:-------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|     `data_dir`     |               | Absolute path to directory where Aquarius data package is stored                                                                                                                 |
|   `archive_dir`    |               | Absolute path to directory where SDS archive format files are to be saved                                                                                                        |
|    `subfolders`    |               | Comma-separated list of direct child directories in data_dir to be processed                                                                                                     |
|     `channels`     |               | Comma-separated list of channels to process (optional). Defaults to only seismometer and hydrophone. Will recognize "all" to process all channels present in the data directory. |
|      `start`       |               | Start date/time for data processing, as YYYYMMDD[HH[MM[SS]]]                                                                                                                     |
|       `end`        |               | End date/time for data processing, as YYYYMMDD[HH[MM[SS]]]                                                                                                                       |
| `correct_metadata` |     False     | Flag to correct channel IDs in output files                                                                                                                                      |
|     `network`      |     `XX`      | FDSN network code assigned to the data. Default `XX` for test data.                                                                                                              |
|  `relative_paths`  |     False     | Specify all paths relative to `data_dir`, with the exception of `archive_dir` and `log_dir`                                                                                      |
|     `metadata`     |               | Path to metadata file (dataless SEED or StationXML). Channel IDs should match the raw data.                                                                                      |
|    `channelmap`    |               | Spreadsheet or delimited text file mapping correct SEED codes to existing identifiers in raw data (same as for QC script). Optional                                              |
|    `extra_meta`    |               | Optional JSON file, as used for QC script. Channel descriptions are taken from here if present.                                                                                  |
|     `log_dir`      |               | Absolute path to directory where runtime logs are to be saved                                                                                                                    |
|     `datalog`      |               | Deployment summary spreadsheet, same as used for QC script                                                                                                                       |
|   `logdelimiter`   |       ,       | Optional delimiter if `datalog` file is delimited text (default comma-separated)                                                                                                 |
|  `legacylogcols`   |     False     | If true, use legacy column names for `datalog` file (opposite behaviour to `logcolnames` in QC script). Will be deprecated in future.                                            |
|   `auxseparate`    |     False     | If true, save output data files for auxiliary channels in a separate SDS folder structure, named the same as `archive_dir` with '_AUX' suffix                                    |
|      `debug`       |     False     | Set logging level to debug for extra information                                                                                                                                 |

### Common Usage

This script is most often used as part of a batch process for data collected from an entire array of OBS (`SDS_many.py`). A JSON file is used as input for the batch script, with an item named `instruments` to specify station-specific parameters. Common CLI parameters are specified at the top level, with boolean flags given as a list labeled `flags`.

Common parameters normally included:
- data_dir (project directory on disk)
- archive_dir ("SDS" optionally in project directory)
- channels ("all")
- channelmap
- extra_meta
- datalog
- network
- flags
  - correct_metadata
  - relative_paths
  - auxseparate

Station-specific parameters typically used:
- subfolders
- metadata
- start (touchdown at seafloor)
- end (release from seafloor)

## Correct and Complete StationXML (`edit_stationxml.py`)

The Aquarius OBS produce StationXML files which include all response information for channels available on the instrument. Values for location may be set during programming of the instrument, but these are generally not known accurately prior to deployment. This script updates all relevant values from the project/station metadata, and corrects SEED codes as necessary to align with the final pre-processed data files.

The StationXML files produced by this script should be compatible with SeisComP, but include only a single station at this time. Automatically combining into a full-network StationXML when run for several input files is a planned future development.
