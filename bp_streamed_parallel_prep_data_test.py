##########################################################################
# ADD SOME GENERAL INFO and LICENSE -> @ajay6763
##########################################################################
from __future__ import division
import sys,os,time
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
#import pygmt
import csv
import pandas as pd
import bp_lib
from joblib import Parallel, delayed
import obspy.geodetics

num_cores = 12
root_order = 2
corr_window=10
snr_window=10
extra_label='_test'
try:
    file = sys.argv[1]
    print('Input file is:',(file))
except:
    print('You did not provided input file (.csv file). Aborting.') 
    exit()
#file='input_EU_7.7.csv'
input = pd.read_csv('./'+file,header=None)
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
azimuth_min=float(res['azimuth_min'])
azimuth_max=float(res['azimuth_max'])
try:
    backazimuth_min=float(res['backazimuth_min'])
    backazimuth_max=float(res['backazimuth_max'])
except:
    pass
dist_min=float(res['dist_min'])
dist_max=float(res['dist_max'])
origin_time=obspy.UTCDateTime(int(res['origin_year']),int(res['origin_month']),
             int(res['origin_day']),int(res['origin_hour']),int(res['origin_minute']),float(res['origin_seconds']))
print(origin_time)

Focal_mech = dict(strike=float(res['event_strike']), dip=float(res['event_dip']), rake=float(res['event_rake'])
                  , magnitude=float(res['event_magnitude']))
stations = str(res['stations'])
waveforms= str(res['waveforms']) 
##########################################################################
# BP parameters
##########################################################################
model               = TauPyModel(model=str(res['model']))
Start_P_cut_time    = float(res['Start_P_cut_time'])  #before P arrival in seconds
End_P_cut_time      = float(res['End_P_cut_time']) #After P arrival seconds
sps                 = float(res['sps'])  #samples per seconds
threshold_correlation=float(res['threshold_correlation'])
SNR=float(res['SNR'])
bp_l                = float(res['bp_l']) #Hz
bp_u                = float(res['bp_u'])   #Hz
stack_start         = int(res['stack_start'])   #in seconds
stack_end           = int(res['stack_end'])  #in seconds
STF_start           = int(res['STF_start'])
STF_end             = int(res['STF_end'])
#smooth_time_window  = int((STF_end-STF_start)/10) #int(res['smooth_time_window'])   #seconds
smooth_time_window  = int(res['smooth_time_window'])   #seconds
smooth_space_window  = int(res['smooth_space_window'])   #seconds
source_grid_size    = float(res['source_grid_size']) #degrees
source_grid_extend  = float(res['source_grid_extend'])   #degrees
source_depth_size   = float(res['source_depth_size']) #km
source_depth_extend = float(res['source_grid_extend']) #km
path = os.getcwd()
#Exp_name=res['Exp_name']
Exp_name=str(Array_name)+'_'+str(event_depth)+'km_'+str(res['model'])+'_'+str(res['threshold_correlation'])+'_corr_'+str(source_grid_size)+'_grid'+str(extra_label)
outdir = str(Event)+'_'+str(Exp_name)
print('Working in Exp:',outdir)
isExist = os.path.exists(outdir)
if not isExist:
        print('\n###########################################')
        print('Output directory does not exist. Making one for you.')
        print('\n###########################################')
        os.makedirs(outdir)
else:
        print('\n###########################################') 
        print('Output directory exists. It will be overwritted.')
        print('\n###########################################')

##########################################################################
# saving the input file 
with open(outdir+'/'+'input.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for row in res.items():
        writer.writerow(row)
##########################################################################
#Array_name           = 'AU'
#inv                  = obspy.read_inventory(stations)
stream_orig          = obspy.read(waveforms)
stream_work          = stream_orig.copy()
##########################################################################
# Making potential sources grid
##########################################################################
slong,slat          = bp_lib.make_source_grid(event_long,event_lat,source_grid_extend,source_grid_size)
##########################################################################
# Load stations inventory
##########################################################################
# Loading the station inventory and data for AU network
# Getting stations name, lat, and longs in a list
# This is done to make a lookup such that when I read the traces
# I can figure out the locations of those traces.
# What I do not understand is why the station and event is not
# included into the trace.stats, like we have in sac.
# May be I am missing something when I download or
# there is clever way or function do this.
sta_net             = []
sta_name            = []
sta_lat             = []
sta_long            = []
sta_dist            = []
sta_azimuth         = []
sta_backazimuth     = []
sta_P_arrival_taup  = []
stations = pd.read_csv(stations, sep='|')
sta_net             = list(stations['Net'])
sta_name            = list(stations['Station'])
sta_lat             = list(stations['Latitude'])
sta_long            = list(stations['Longitude'])
sta_dist            = list(stations['Distance'])
sta_azimuth         = list(stations['Azimuth'])
print('Total number of stations:', len(sta_lat))

##########################################################################
# reading wavefrom data and assigning station info to them 
# I do this by haveing lists of stations with all the info read and 
# extracted above and then look-up for the station name in the waveform
# and doing the assignment
##########################################################################
# Looping through the network traces and writing 
# station latitude and station longitude 
sta_sps=[]
for t in stream_work:
        sta          = t.stats.station
        #net 
        if sta in sta_name:
            ind                          = sta_name.index(sta)
            t.stats['Dist']              = sta_dist[ind]
            t.stats['Azimuth']           = sta_azimuth[ind]
            ## look for documentation of gps2dist_azimuth
            #baz = gps2dist_azimuth(event_lat, event_long, sta_lat[ind], sta_long[ind])
            #t.stats['Backazimuth']       =  baz[2]
            t.stats['station_latitude']  = sta_lat[ind]
            t.stats['station_longitude'] = sta_long[ind]
            t.stats['origin_time']       = origin_time
            #arrivals                     = model.get_travel_times(source_depth_in_km=event_depth,distance_in_degree=locations2degrees(event_lat,event_long,sta_lat[ind],sta_long[ind]),phase_list=["P"])
            arrivals                     = model.get_travel_times(source_depth_in_km=event_depth,distance_in_degree=t.stats.Dist,phase_list=["P"])
            arr                          = arrivals[0]
            t_travel                     = arr.time;
            t.stats['P_arrival']         = origin_time + t_travel 
            sta_sps.append(t.stats.sampling_rate)
        else:
            stream_work.remove(t)
print("Total no stations with data:", len(stream_work))
print("Sampling rate of the waveform data:", np.unique(sta_sps))
################################
## Bound selection
##########################################################################
# SPS and distance check and azimuth
print('Total no of traces before decimation criteria:', len(stream_work))
stream_SPS = bp_lib.check_sps(stream_work,sps)
print('Total no of traces after decimation criteria:', len(stream_SPS))
######### distance
print('Total no of traces before  distance criteria:', len(stream_SPS))
stream_SPS_dist = bp_lib.check_distance(stream_SPS,dist_min,dist_max)
print('Total no of traces after distance criteria:', len(stream_SPS_dist))

######### azimuth
print('Total no of traces before  azimuth criteria:', len(stream_SPS_dist))
stream_SPS_dist_azimuth=stream_SPS_dist.copy()
stream_SPS_dist_azimuth = bp_lib.check_azimuth(stream_SPS_dist_azimuth,azimuth_min,azimuth_max)
print('Total no of traces after azimuth criteria:', len(stream_SPS_dist_azimuth))
'''
######### baz
print('Total no of traces before  azimuth criteria:', len(stream_SPS_dist))
stream_SPS_dist_azimuth=stream_SPS_dist.copy()
stream_SPS_dist_azimuth = bp_lib.check_baz(stream_SPS_dist_azimuth,backazimuth_min,backazimuth_max)
print('Total no of traces after backazimuth criteria:', len(stream_SPS_dist_azimuth))
'''
#############################
# Except selection
######### azimuth
#print('Total no of traces before  azimuth criteria:', len(stream_SPS_dist))
#stream_SPS_dist_azimuth = bp_lib.check_azimuth_except(stream_SPS_dist,azimuth_min,azimuth_max)
#print('Total no of traces after azimuth criteria:', len(stream_SPS_dist_azimuth))

######### SNR
stream_SPS_dist_azimuth_SNR=stream_SPS_dist_azimuth.copy()
print('Total no of traces before  SNR criteria:', len(stream_SPS_dist_azimuth_SNR))
stream_SPS_dist_azimuth_SNR = bp_lib.snr_check(stream_SPS_dist_azimuth_SNR,SNR,snr_window,snr_window)
print('Total no of traces after SNR criteria:', len(stream_SPS_dist_azimuth_SNR))
'''
## get the displacement
for tr in stream_SPS_dist_azimuth_SNR:
    tr.detrend("linear")
    tr.integrate()
    #tr.filter('bandpass',freqmin=bp_l,freqmax=bp_u,corners=5)
    tr.detrend("linear")
'''
##########################################################################
# finding reference station
# the the station in the middle of the array
# I will take the mean of the distance and azimuth and take the station closest to it
##########################################################################
stream_correlated=stream_SPS_dist_azimuth_SNR.copy()
Ref_station_index=bp_lib.get_ref_station(stream_correlated)
ref_trace = stream_SPS_dist_azimuth_SNR[Ref_station_index]
print(Ref_station_index)
print(ref_trace)
print('Total no of traces before Cross-correlation:', len(stream_correlated))
#print('Performning cross-correlation by bandpass filtering at bp_l = %f and bp_u = %f' %(bp_l,bp_u))
print('Performning cross-correlation')
stream_correlated=bp_lib.crosscorr_stream_xcorr(stream_correlated,ref_trace,corr_window,bp_l,bp_u)
print('Total no of traces after Cross-correlation:', len(stream_correlated))
##########################################################################
# Selecting traces for BP. 
# Use can choose correlation threshold
# If not then simple set threshold to 0
# and all the traces will be used for in
# back-projection
##########################################################################
#stream_for_bp = stream_cut_filtered_correlated.copy()
stream_for_bp_corr_thresh = stream_correlated.copy()
print('No of traces before cross-correlation threshold = ', len(stream_for_bp_corr_thresh))
for tr in stream_for_bp_corr_thresh:
    if (abs(tr.stats.Corr_coeff) >= threshold_correlation):
        #time = np.arange(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.delta)
        #plt.plot(time,tr.data/np.max(tr.data))
        pass
    else:
        stream_for_bp_corr_thresh.remove(tr)
        #pass
print('No of traces after cross-correlation threshold = ', len(stream_for_bp_corr_thresh))
#bp_lib.data_plot(stream_for_bp_corr_thresh,event_long,event_lat,outdir,'data_plot_'+str(Array_name)+'.png')
##########################################################################
# Filtering traces
##########################################################################
#stream_filtered=stream_for_bp_corr_thresh.copy()
#for tr in stream_filtered:
#    #tr.detrend
#    tr.filter('bandpass',freqmin=bp_l,freqmax=bp_u,corners=5)
#    tr.detrend("linear")
    #tr.normalize
#stream_cut=stream_filtered.copy()
#temp=stream_filtered.copy()
#print('Total no of traces before STA-LTA:', len(temp))
#temp=bp_lib.STA_LTA(temp,20,10,Start_P_cut_time)
#print('Total no of traces after Cross-correlation:', len(temp))
##########################################################################
# CUtting before and after P arrival 
##########################################################################
stream_cut=stream_for_bp_corr_thresh.copy()
print('Total no of traces before data gap checks:', len(stream_cut))
for t in stream_cut:
        #t.trim(t.stats['P_arrival']-Start_P_cut_time,t.stats['P_arrival']+End_P_cut_time)
        t.trim(t.stats['P_arrival']-Start_P_cut_time,t.stats['P_arrival']+End_P_cut_time)
        if t.stats.npts < (Start_P_cut_time+End_P_cut_time)/t.stats.delta:
            stream_cut.remove(t)
        else:
            pass
#stream_for_bp = bp_lib.snr_check(stream_cut,2,5,50)
print('Total no of traces after cutting and data gap checks and final no of traces for bp:', len(stream_cut))
stream_for_bp=stream_cut.copy()
#Ref_station_index=bp_lib.get_ref_station(stream_cut_correlated,'mean','')
Ref_station_index=bp_lib.get_ref_station(stream_for_bp)
ref_trace = stream_for_bp[Ref_station_index]
print('Reference station')
print(Ref_station_index)
print(ref_trace)
#stream_cut_filtered_correlated=bp_lib.crosscorr_stream(stream_cut_filtered,ref_trace,5)
#stream_correlated = stream_SPS_dist_azimuth_SNR.copy()
#print('Total no of traces before Cross-correlation:', len(stream_cut_correlated))
#stream_cut_correlated=bp_lib.polarity(stream_cut_correlated,1/bp_l)
#stream_cut_correlated=bp_lib.crosscorr_stream_xcorr(stream_cut_correlated,ref_trace,10)
#print('Total no of traces after Cross-correlation:', len(stream_cut_correlated))
# recalculating station weight
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
#bp_lib.data_plot(stream_for_bp_corr_thresh,event_long,event_lat,outdir,'data_plot_'+str(Array_name)+'.png')
##########################################################################
# doing the back-projection
# beam variable is the "beast". 
# This will be an array who has size = [len[slat], len(stack_window)]
# Basically this contains the traces from all the stations from an array 
# that are alligned according to the arrival-time of the P-wave from the each source
# grid point.  
##########################################################################
time_start = time.process_time()
print('Finished preparing data.')
print('Now writing the stream and its info.')
def process_location(j, slat, slong, stream_for_bp, event_depth, origin_time, stack_start, stack_end):
    source_stream_info = []
    for t in stream_for_bp:
        distance = obspy.geodetics.locations2degrees(slat[j], slong[j], t.stats.station_latitude, t.stats.station_longitude)
        arrivals = model.get_travel_times(source_depth_in_km=event_depth, distance_in_degree=distance, phase_list=["P"])
        arr = arrivals[0]
        t_travel = arr.time
        t_total = origin_time + t_travel #+ t.stats.Corr_shift
        source_stream_info.append([slat[j], slong[j],t.stats.station,t_total])
    #stream_temp.write(outdir+'/'+str(j)+"_.mseed")
    #bp_info = bp_lib.save_stream_info(stream_for_bp)
    #np.save(outdir+'/'"array_bp_info",bp_info,allow_pickle=True)
    #print("Progress =",np.round(((j/len(slat))*100)),"%" )
    #stack_sum = bp_lib.nth_root_stacking(stack_reshaped, 4)
    return source_stream_info
    #return np.power(np.mean(np.power(stack_reshaped, root_order), axis=0), 1/root_order)

#print('Writing the stream info in parallel.')
#results = Parallel(n_jobs=num_cores)(
#    delayed(process_location)(j, slat, slong, stream_for_bp, event_depth, origin_time, stack_start, stack_end)
#    for j in range(len(slat)) )
beam=[];
for j in range(len(slat)):
    source_stream_info=[]
    for t in stream_for_bp:
        distance = obspy.geodetics.locations2degrees(slat[j], slong[j], t.stats.station_latitude, t.stats.station_longitude)
        arrivals = model.get_travel_times(source_depth_in_km=event_depth, distance_in_degree=distance, phase_list=["P"])
        arr = arrivals[0]
        t_travel = arr.time
        t_total = origin_time + t_travel #+ t.stats.Corr_shift
        source_stream_info.append([slat[j], slong[j],t.stats.station,t_total])
    beam = np.concatenate(source_stream_info)
np.save(outdir+'/'"beam_info",beam,allow_pickle=True)
print('Writing the array info in parallel.')
bp_info = bp_lib.save_stream_info(stream_for_bp)
np.save(outdir+'/'"array_bp_info",bp_info,allow_pickle=True)
print('Writing the stream.')
stream_for_bp.write(outdir+'/'"stream.mseed")

#beam_reshaped = beam.reshape((len(slat), (stack_end-stack_start)*int(sps)))
print("Progress back-projection DONE for Exp:",  outdir)
#print("Progress back-projection DONE!!")
print('Total time taken:',time.process_time() - time_start)