import argparse
from datetime import datetime
from glob import glob
import obspy
import os
import re
import timeit
import traceback
import warnings

from obspy.clients.fdsn import Client
from obspy.core.event import Catalog
from obspy.core.stream import Stream
from obspy.geodetics.base import gps2dist_azimuth
from obspy.geodetics import kilometer2degrees

from obstools.atacr import utils
import stdb
import nfsi_obs as nf

# Ensure resource directory exists
resource_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'resource/OBSDataPipeline')
if not os.path.isdir(resource_dir):
    os.makedirs(resource_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare seismic and hydrophone data for tilt and compliance '
                                                 'correction processing with OBSTools package. Assumes data has been '
                                                 'saved as day-long miniSEED files.')
    parser.add_argument('indb', help="Station Database to process from.", type=str)
    parser.add_argument('--base_dir', dest='base_dir', help="Base directory where all files are stored (or will be "
                                                            "specified relative to).")
    parser.add_argument('--relative_paths', dest="relative_paths", action="store_true",
                        help="If true, all other path arguments are specified relative to the base directory.")
    parser.add_argument('--outdir', dest="outdir",
                        help="Output directory, if different from base directory.")
    parser.add_argument('--O', dest="overwrite", action="store_true", help="Overwrite existing output data files.")

    ServerGroup = parser.add_argument_group(title="Server Settings", description="Settings for which datacenter to login to.")
    ServerGroup.add_argument('-S', '--server', dest="server", type=str, default="IRIS",
                             help="Specify the server to connect to. Options include: BGR, ETH, GEONET, GFZ, INGV, "
                                  "IPGP, IRIS, KOERI, LMU, NCEDC, NEIP, NERIES, ODC, ORFEUS, RESIF, SCEDC, USGS, USP. "
                                  "[Default IRIS]")

    EventGroup = parser.add_argument_group(title="Event Settings",
                                           description="Settings for refining the event catalogue")
    EventGroup.add_argument('--start', dest="startdate", help="Start date of time period to be analyzed, as YYYYMMDD")
    EventGroup.add_argument('--end', dest="enddate", help="End date of time period to be analyzed, as YYYYMMDD")
    EventGroup.add_argument('--min-mag', type=float, dest='minmag', default=5.5,
                            help="Specify the minimum magnitude of event for which to search. [Default 5.5]")
    EventGroup.add_argument('--max-mag', type=float, dest='maxmag', default=None,
                            help="Specify the minimum magnitude of event for which to search. [Default no limit]")

    GeomGroup = parser.add_argument_group(title="Geometry Settings",
                                          description="Settings associated with the event-station geometries.")
    GeomGroup.add_argument('--min-dist', type=float, dest='mindist', default=30.,
                           help="Specify the minimum great circle distance (degrees) between the station and event. "
                                "[Default 30]")
    GeomGroup.add_argument('--max-dist', type=float, dest='maxdist', default=120.,
                           help="Specify the maximum great circle distance (degrees) between the station and event. "
                                "[Default 120]")

    FreqGroup = parser.add_argument_group(title="Frequency Settings")
    FreqGroup.add_argument('--window', type=float, dest="window", default=7200.,
                           help="Specify window length for event data in seconds. Default value is highly recommended. "
                                "Program may not be stable for large deviations from default value. [Default 7200 (2 "
                                "hours)]")

    try:
        start_time = datetime.now()
        t0 = timeit.default_timer()
        args = parser.parse_args()
        if not os.path.exists(args.indb):
            parser.error('Input file {} does not exist'.format(args.indb))

        if args.base_dir:
            base_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.base_dir)))
        else:
            base_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(os.path.join(resource_dir, 'test_data'))))

        tstart, tend = None, None
        if args.startdate:
            tstart = datetime.strptime(args.startdate, '%Y%m%d')
        if args.enddate:
            tend = datetime.strptime(args.enddate, '%Y%m%d')

        db = stdb.io.load_db(fname=args.indb)
        allkeys = db.keys()
        sorted(allkeys)

        for stkey in allkeys:
            sta = db[stkey]
            data_dir = os.path.join(base_dir, 'DATA', stkey)
            dataout = os.path.join(base_dir, 'EVENTS', stkey)
            if not os.path.exists(dataout):
                os.makedirs(dataout)

            datafiles = glob(os.path.join(data_dir, '*.SAC'))
            full_data = Stream()
            for df in datafiles:
                temp = obspy.read(df)
                for tr in temp:
                    full_data.append(tr)
            full_data.merge()

            # Establish FDSN client
            client = Client(args.server)

            # Start and end times for catalog search
            if tstart is None:
                tstart = sta.startdate
            if tend is None:
                tend = sta.enddate
            if tstart > sta.enddate or tend < sta.startdate:
                continue

            # Get catalog events
            cat = client.get_events(starttime=tstart, endtime=tend, minmagnitude=args.minmag, maxmagnitude=args.maxmag)
            nev = len(cat)
            print('Found {} possible events'.format(len(cat)))

            sta_cat = Catalog()

            for ev in cat:
                time = ev.origins[0].time
                lat = ev.origins[0].latitude
                lon = ev.origins[0].longitude
                dep = ev.origins[0].depth
                epi_dist, az, baz = gps2dist_azimuth(lat, lon, sta.latitude, sta.longitude)
                epi_dist /= 1000.
                gcd = kilometer2degrees(epi_dist)
                mag = ev.magnitudes[0].mag
                if mag is None:
                    mag = -9.

                # Display event info
                print('Origin time: {0} | Lat: {1:6.2f}, Lon: {2:7.2f}, Dep: {3:6.2f}, Mag: {4:3.1f} | Dist: {5:7.2f} km; {6:7.2f} deg'.format(time.strftime('%Y-%m-%d %H:%M:%S'), lat, lon, dep, mag, epi_dist, gcd))

                if not (gcd > args.mindist and gcd < args.maxdist):
                    print('    -> Event outside epicentral distance range - skipping')
                    continue

                sta_cat.append(ev)

                tstamp = str(time.year).zfill(4) + '.' + str(time.julday).zfill(3) + '.' + str(time.hour).zfill(
                    2) + '.' + str(time.minute).zfill(2)
                t1 = time
                t2 = t1 + args.window

                evt_data = full_data.slice(t1, t2, nearest_sample=False)
                evt_data = evt_data.split()    # Just in case one of the traces is masked... not sure why this happens
                for tr in evt_data:
                    fileout = os.path.join(dataout, tstamp+'.'+tr.stats.channel+'.SAC')
                    tr.write(fileout, format='SAC')

            print('Found {} events which fit criteria'.format(len(sta_cat)))

        end_time = datetime.now()
        run_time = timeit.default_timer() - t0
        if run_time < 60:
            print("Total run time: {0} seconds".format(run_time))
        elif run_time < 3600:
            print("Total run time: {0} seconds ({1} minutes)".format(run_time, run_time/60))
        elif run_time < 3600*24:
            print("Total run time: {0} seconds ({1} hours)".format(run_time, run_time/3600))
        else:
            print("Total run time: {0} seconds ({1} days)".format(run_time, run_time/3600/24))

    except Exception as e:
        print(traceback.print_exc())
        parser.print_help()
        exit(1)
