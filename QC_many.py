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
    ['AQU-1B61'],
    ['AQU-1F61'],
    ['AQU-2A61'],
    ['AQU-2B62'],
    ['AQU-2C62'],
    ['AQU-3B61', '20220715'],
    ['AQU-0560'],
    ['AQU-0660'],
    ['AQU-755C', '20220715'],
    ['AQU-1561', '20220715'],
    ['AQU-2962'],
    ['AQU-3661'],
    ['AQU-BA5C'],
    ['AQU-BA60', '20220715'],
    ['AQU-BB60', '20220714'],
    ['AQU-BC60', '20220715'],
    ['AQU-BD60', '20220715'],
    ['AQU-BE60', '20220714'],
    ['AQU-C05E'],
    ['AQU-C060', '20220714'],
    ['AQU-C560', '20220714'],
    ['AQU-C760', '20220714'],
    ['AQU-CB5A'],
    ['AQU-CB60', '20220715'],
    ['AQU-CD60', '20220715'],
    ['AQU-D261'],
    ['AQU-D361'],
    ['AQU-DA61'],
    ['AQU-E361'],
    ['AQU-E661', '20220715'],
]

time_info = []

for inst in instruments:
    istart = timeit.default_timer()

    args_list = [
        'python',
        'OBS_QC.py',
        '--config=~/resource/OBSDataPipeline/configs/QC_config_Batch3-4_intake.ini',
        '--data_dir={}'.format(inst[0]),
        '--obsid={}'.format(inst[0]),
        '--debug'
    ]
    if len(inst) > 1:
        args_list.append('--start={}'.format(inst[1]))

    subprocess.run(args_list)

    iend = timeit.default_timer()
    time_info.append([inst[0], iend - istart])

full_end = timeit.default_timer()

print('Runtime by instrument:')
for ti in time_info:
    print('{0}: {1:.3f} seconds ({2:.3f} minutes)'.format(ti[0], ti[1], ti[1] / 60))

print('Total runtime: {0:.3f} seconds ({1:.3f} minutes)'.format(full_end - full_start, (full_end - full_start) / 60))
