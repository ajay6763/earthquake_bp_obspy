import sys,os
import obspy
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.trigger import recursive_sta_lta_py
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib.transforms as mtransforms
import pygmt
import csv
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import time
import multiprocessing as mp
########### 
import bp_lib
import math
def moving_average(x, w):
    """
    Computes the moving average of a 2D numpy array x with a window size of w.
    """
    return np.convolve(x, np.ones(w), 'same') / w


name='EAF_7.7_JP_10.0km_ak135_0.5_grid_3_extend'
path = os.getcwd()
outdir = name #str(Event)+'_'+str(Exp_name)
bp_l=1
bp_u=5
smooth_time_window=10
space_window=1
peak_scale=10
stack_start=30
stack_end=200
STF_start=0
STF_end=150


#extra_label='_BB'
#stack_end=100

input = pd.read_csv('./'+name+'/input.csv',header=None)
a=input.to_dict('series')
keys = a[0][:]
values = a[1][:]
res = {}
for i in range(len(keys)):
        res[keys[i]] = values[i]
        #print(keys[i],values[i])
##########################################################################
# Event info
Event=res['Event']
event_lat=float(res['event_lat'])
event_long=float(res['event_long'])
event_depth=float(res['event_depth'])
Array_name=res['Array_name']
Exp_name=res['Exp_name']
azimuth_min=float(res['azimuth_min'])
azimuth_max=float(res['azimuth_max'])
dist_min=float(res['dist_min'])
dist_max=float(res['dist_max'])
origin_time=obspy.UTCDateTime(int(res['origin_year']),int(res['origin_month']),
             int(res['origin_day']),int(res['origin_hour']),int(res['origin_minute']),float(res['origin_seconds']))
print(origin_time)
Focal_mech = dict(strike=float(res['event_strike']), dip=float(res['event_dip']), rake=float(res['event_rake'])
                 , magnitude=float(res['event_magnitude']))
model               = TauPyModel(model=str(res['model']))
sps                 = int(res['sps'])  #samples per seconds
threshold_correlation=float(res['threshold_correlation'])
SNR=float(res['SNR'])
#smooth_time_window  = int((stack_end-stack_start)/10) #int(res['smooth_time_window'])   #seconds
source_grid_size    = float(res['source_grid_size']) #degrees
source_grid_extend  = float(res['source_grid_extend'])   #degrees
source_depth_size   = float(res['source_depth_size']) #km
source_depth_extend = float(res['source_grid_extend']) #km

#stream_for_bp=obspy.read('./Turky_7.6_all/stream.mseed')
slong,slat          = bp_lib.make_source_grid(event_long,event_lat,source_grid_extend,source_grid_size)



stations_file = str(res['stations'])
stream_for_bp= obspy.read('./'+name+'/stream.mseed') 
beam_info = np.load('./'+name+'/beam_info.npy',allow_pickle=True)
stream_info = np.load('./'+name+'/array_bp_info.npy',allow_pickle=True)
print('#############################################################################\n')
print('Exp:',name)
print('Origin time:',origin_time)
print('Long= %f Lat= %f Depth= %f' % (event_long,event_lat,event_depth))
print('bp_low= %f bp_high= %f Correlation threshold= %f SNR= %f'% (bp_l,bp_u,threshold_correlation,SNR))
print('#############################################################################\n')

print('Done loading data')




sta_name=list(stream_info[:,1])
for t in stream_for_bp:
        if len(t.stats['station'].split('.')) > 1:
            sta          = t.stats.station+str('H')
        else:
            sta          = t.stats.station
        #net 
        if sta in sta_name:
            ind                          = sta_name.index(sta)
            t.stats['origin_time']       = origin_time
            t.stats['station_longitude'] = float(stream_info[ind,2])
            t.stats['station_latitude']  = float(stream_info[ind,3])
            t.stats['Dist']              = float(stream_info[ind,4])
            t.stats['Azimuth']           = float(stream_info[ind,5])
            arrivals                     = model.get_travel_times(source_depth_in_km=event_depth,distance_in_degree=t.stats.Dist,phase_list=["P"])
            arr                          = arrivals[0]
            t_travel                     = arr.time;
            t.stats['P_arrival']         = origin_time + t_travel +  timedelta(hours=9)
        
            #t.stats['P_arrival']         = float(stream_info[ind,6]) 
            t.stats['Corr_coeff']        = float(stream_info[ind,7])
            t.stats['Corr_shift']        = float(stream_info[ind,8])
            t.stats['Corr_sign']         = float(stream_info[ind,9])
        else:
            pass
            #print('Something is not right.')


Ref_station_index=bp_lib.get_ref_station(stream_for_bp)
ref_trace = stream_for_bp[Ref_station_index]

for tr in stream_for_bp:
    count=0;
    for tr_ in stream_for_bp:
        dist=((tr.stats.station_latitude-tr_.stats.station_latitude)**2 + 
              (tr.stats.station_longitude-tr_.stats.station_longitude)**2 )**0.2;
        if ( dist <= 1):
            count=count+1;
        else:
            continue
    tr.stats['Station_weight'] = count




##########################################################################
# Make beam
beam_info_reshaped=beam_info.reshape(len(slat),len(stream_for_bp),4)
time_start = time.process_time()
beam=[] #obspy.Stream()
for j in range(len(beam_info_reshaped)):
    source = beam_info_reshaped[j]
    stream_source=stream_for_bp.copy()
    for i in range(len(source)):
        tr = stream_source.select(station=source[i][2])
        arrival=source[i][3]+tr[0].stats.Corr_shift
        tr.trim(arrival-stack_start,arrival+stack_end)
        tr.detrend('linear')
        tr.normalize()
    stream_use=stream_source.copy()
    #stack=stream_use.stack('linear')
    stack=[]
    for tr in stream_use:
        tr.filter('bandpass',freqmin=bp_l,freqmax=bp_u,corners=5)
        tr.detrend("linear")
        cut = tr.data * tr.stats.Corr_coeff/tr.stats.Station_weight
        #tr.normalize()
        stack.append(cut[0:int((stack_start+stack_end)*sps)])
    #print(np.shape(stack))
    #stack_reshaped = np.array(stack).reshape((len(stream_for_bp), -1))
    #beam.append(np.sum(stack_reshaped,axis=0))
    beam.append(np.sum(stack,axis=0))
#do some stuff
print('Total time taken:',time.process_time() - time_start)
file_save='beam_'+str(bp_l)+'_'+str(bp_u)+'_'+str(Array_name)+'.dat'
np.savetxt(outdir+'/'+file_save,beam)

def process_beam(j):
    source = beam_info_reshaped[j]
    stream_source=stream_for_bp.copy()
    for i in range(len(source)):
        tr = stream_source.select(station=source[i][2])
        arrival=source[i][3]+tr[0].stats.Corr_shift
        tr.trim(arrival-stack_start,arrival+stack_end)
        tr.detrend('linear')
        tr.normalize()
    stream_use=stream_source.copy()
    stack=[]
    for tr in stream_use:
        tr.filter('bandpass',freqmin=bp_l,freqmax=bp_u,corners=5)
        tr.detrend("linear")
        cut = tr.data * tr.stats.Corr_coeff/tr.stats.Station_weight
        stack.append(cut[0:int((stack_start+stack_end)*sps)])
    return np.sum(stack,axis=0)

if __name__ == '__main__':
    # Make beam
    beam_info_reshaped=beam_info.reshape(len(slat),len(stream_for_bp),4)
    time_start = time.process_time()
    beam=[] #obspy.Stream()
    with mp.Pool() as pool:
        results = pool.map(process_beam, range(len(beam_info_reshaped)))
        beam = [r for r in results]
    print('Total time taken:',time.process_time() - time_start)
    file_save='beam_'+str(bp_l)+'_'+str(bp_u)+'_'+str(Array_name)+'.dat'
    np.savetxt(outdir+'/'+file_save,beam)


