from copy import copy, deepcopy


def cut_trace(tr, starttime=None, endtime=None, nearest_sample=True, pad=False, fill_value=None):
    trc = copy(tr)
    trc.stats = deepcopy(tr.stats)
    trc.trim(starttime=starttime, endtime=endtime, pad=pad, nearest_sample=nearest_sample, fill_value=fill_value)
    return trc
