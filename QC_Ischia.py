"""
Run field data QC script (OBS_QC.py) for several instruments sequentially. This is especially useful for dry-land test
deployments, when many instruments will be "recovered" in a short time and data volumes per instrument are relatively
low (a few days to up to a few months normally).

Author: K. Bosman
June 1, 2023
"""
import subprocess
import timeit

full_start = timeit.default_timer()

instruments = [
    {
        'obsid': 'AQU-4261',
        'data_dir': 'AQU-4261',
        'metadata': 'AQU-4261/DG_04261_fudge.dataless'
    },
    {
        'obsid': 'AQU-8063',
        'data_dir': 'AQU-8063-fixed',
        'metadata': 'AQU-8063-fixed/DG_08063_fudge.dataless'
    },
    {
        'obsid': 'AQU-8263',
        'data_dir': 'AQU-8263',
    },
    {
        'obsid': 'AQU-B063',
        'data_dir': 'AQU-B063',
    },
]

time_info = []

for inst in instruments:
    istart = timeit.default_timer()

    args_list = [
        'python',
        'OBS_QC.py',
        '--config=~/resource/OBSDataPipeline/configs/QC_config_Ischia2023.ini',
        '--colormap=seismic',
        '--debug'
    ]
    for key in inst:
        args_list.append('--{0}={1}'.format(key, inst[key]))

    subprocess.run(args_list)

    iend = timeit.default_timer()
    time_info.append([inst['obsid'], iend - istart])

full_end = timeit.default_timer()

print('Runtime by instrument:')
for ti in time_info:
    print('{0}: {1:.3f} seconds ({2:.3f} minutes)'.format(ti[0], ti[1], ti[1] / 60))

print('Total runtime: {0:.3f} seconds ({1:.3f} minutes)'.format(full_end - full_start, (full_end - full_start) / 60))
