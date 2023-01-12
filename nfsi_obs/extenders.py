from copy import copy, deepcopy


def cut_trace(tr, starttime=None, endtime=None, nearest_sample=True, pad=False, fill_value=None):
    """
    Return a section of the input trace between `starttime` and `endtime`, without modifying the input trace object.
    Obspy's Trace.trim() method modifies the data array in-place (no copying), and Trace.slice() does not allow padding.
    All input parameters other than the trace itself are passed directly to `obspy.core.trace.Trace.trim()`

    :param tr: obspy.core.trace.Trace object
    :param starttime: obspy.core.UTCDateTime object
    :param endtime: obspy.core.UTCDateTime object
    :param nearest_sample:
    :param pad:
    :param fill_value:
    :return:
    """
    trc = copy(tr)
    trc.stats = deepcopy(tr.stats)
    trc.trim(starttime=starttime, endtime=endtime, pad=pad, nearest_sample=nearest_sample, fill_value=fill_value)
    return trc
