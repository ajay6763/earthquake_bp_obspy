from __future__ import division
import sys,os
import obspy
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
from obspy.geodetics.base import gps2dist_azimuth
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import bp_lib as bp_lib
import multiprocessing as mp
########### 
time_start = time.time()
try:
    name = sys.argv[1]
    print('Input experiment is :',(name))
except:
    print('You did not provided experiment name (foleder). Aborting.') 
    exit()
#name='KYR_7.0_EU_13.0km_iasp91_0.2_grid_'
path = os.getcwd()
outdir = name #str(Event)+'_'+str(Exp_name)
input = pd.read_csv('./'+name+'/input.csv',header=None)
a=input.to_dict('series')
keys = a[0][:]
values = a[1][:]
res = {}
for i in range(len(keys)):
        res[keys[i]] = values[i]
## BP parameters from the input file
try:
    bp_l = float(sys.argv[2])
    bp_u = float(sys.argv[3])
    print('bp_l and bp_u is:',(bp_l,bp_u))
except:
    bp_l                = float(res['bp_l']) #Hz
    bp_u                = float(res['bp_u'])   #Hz
smooth_time_window  = int(res['smooth_time_window'])   #seconds
smooth_space_window = int(res['smooth_space_window'])
stack_start         = int(res['stack_start'])   #in seconds
stack_end           = int(res['stack_end'])  #in seconds
STF_start           = int(res['STF_start'])
STF_end             = int(res['STF_end'])
sps                 = int(res['sps'])  #samples per seconds
threshold_correlation=float(res['threshold_correlation'])
SNR=float(res['SNR'])
# Event info
Event=res['Event']
event_lat=float(res['event_lat'])
event_long=float(res['event_long'])
event_depth=float(res['event_depth'])
Array_name=res['Array_name']
#Exp_name=res['Exp_name']
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
source_grid_size    = float(res['source_grid_size']) #degrees
#source_grid_extend  = float(res['source_grid_extend'])   #degrees
source_grid_extend_x  = float(res['source_grid_extend_x'])   #degrees
source_grid_extend_y  = float(res['source_grid_extend_y'])   #degrees
source_depth_size   = float(res['source_depth_size']) #km
#slong,slat          = bp_lib.make_source_grid(event_long,event_lat,source_grid_extend,source_grid_size)
slong,slat          = bp_lib.make_source_grid_hetero(event_long,event_lat,source_grid_extend_x,source_grid_extend_y,source_grid_size)

stations_file = str(res['stations'])
stream_for_bp= obspy.read('./'+name+'/stream.mseed') 
beam_info = np.load('./'+name+'/beam_info.npy',allow_pickle=True)
stream_info = np.load('./'+name+'/array_bp_info.npy',allow_pickle=True)
print('#############################################################################\n')
print('Exp:',name)
print('Origin time:',origin_time)
print('Long= %f Lat= %f Depth= %f' % (event_long,event_lat,event_depth))
print('bp_low= %f bp_high= %f Correlation threshold= %f SNR= %f'% (bp_l,bp_u,threshold_correlation,SNR))
filter='on'
if bp_l<0 or bp_u<0:
    print('NOTE!! : will not filter the waveforms before stacking. The beam will be unfiltered.')
    filter='off'
else:
    filter='on'
print('#############################################################################\n')
print('Done loading data.')
print('Total time taken:',time.process_time() - time_start)
print('Now gathering stream information..')
### removing mean from the traces
stream_for_bp= bp_lib.detrend_normalize_stream(stream_for_bp,type='demean')
stream_for_bp=bp_lib.populate_stream_info(stream_for_bp,stream_info,origin_time,event_depth,model)
Ref_station_index=bp_lib.get_ref_station(stream_for_bp)
ref_trace = stream_for_bp[Ref_station_index]
print('Done gathering stream information.')
print("Time taken: {:.1f} min".format((time.time()-time_start)/60.0))
print('Computing computing station weight.')
stream_for_bp=bp_lib.stream_station_weight(stream_for_bp)
print('Done computing station weight.')
print("Time taken: {:.1f} min".format((time.time()-time_start)/60.0))
print('Now making the beam...')
##########################################################################
# Make beam
'''
beam_info_reshaped=beam_info.reshape(len(slat),len(stream_for_bp),4)
print('beam_info',np.shape(beam_info))
print('beam_info_reshaped',np.shape(beam_info_reshaped))
beam=[] 
for j in range(len(beam_info_reshaped)):
    source = beam_info_reshaped[j]
    stream_source=stream_for_bp.copy()
    for i in range(len(source)):
        tr = stream_source.select(station=source[i][2])
        arrival=source[i][3]+tr[0].stats.Corr_shift
        tr.trim(arrival-stack_start,arrival+stack_end)
    stream_use=stream_source.copy()
    stack=[]
    for tr in stream_use:
        tr.filter('bandpass',freqmin=bp_l,freqmax=bp_u)
        cut = tr.data * tr.stats.Corr_coeff/tr.stats.Station_weight
        stack.append(cut[0:int((stack_start+stack_end)*sps)])
    beam.append(np.sum(stack,axis=0))
    bp_lib.progressbar(j)
print('Done making the beam.')
print("Time taken: {:.1f} min".format((time.time()-time_start)/60.0))
print('Saving the beam.')
file_save='beam_'+str(bp_l)+'_'+str(bp_u)+'_'+str(Array_name)+'.dat'
np.savetxt(outdir+'/'+file_save,beam)
print("Total execution time: {:.1f} min".format((time.time()-time_start)/60.0))
'''
def process_beam(j):
    source = beam_info_reshaped[j]
    stream_source=stream_for_bp.copy()
    for i in range(len(source)):
        tr = stream_source.select(station=source[i][2])
        arrival=source[i][3]+tr[0].stats.Corr_shift #+2.5
        tr.trim(arrival-stack_start,arrival+stack_end)
    stream_use=stream_source.copy()
    stack=[]
    for tr in stream_use:
        if filter=='on':
            tr.filter('bandpass',freqmin=bp_l,freqmax=bp_u)
        else:
            pass
        tr.integrate()
        tr.detrend(type='demean')
        tr.normalize()
        cut = tr.data/np.max(np.abs(tr.data)) * tr.stats.Corr_coeff/tr.stats.Station_weight
        #cut = tr.data #/tr.stats.Station_weight
        #cut = tr.data*tr.stats.Corr_coeff/tr.stats.Station_weight
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
    print("Done makeing the beam for:", outdir)
    print("The frequency band used was:", bp_l,bp_u)
    print("Total time taken: {:.1f} min".format((time.time()-time_start)/60.0))
    print('#########################################################')
    print()
    print("Now to plot run the following: ")  
    print()
    print("python bp_post_process_make_results.py", outdir)  
    print()
    print("You can also pass the frequency band other than in the input file but you should have the beam made for that.")  


