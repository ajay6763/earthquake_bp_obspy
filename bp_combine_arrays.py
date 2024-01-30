##########################################################################
# ADD SOME GENERAL INFO and LICENSE -> @ajay6763
##########################################################################
from __future__ import division
import obspy
from obspy.geodetics import locations2degrees
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.cross_correlation import xcorr_pick_correction # for cross-correlation
from obspy.signal.trigger import recursive_sta_lta_py
from scipy import signal

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib.transforms as mtransforms

########### 
#import bp lib
import bp_lib as bp_lib
array_list=sys.arg(1)
ref_array=array_list[0]
outdir = './combined' #str(Event)+'_'+str(Exp_name)
path = os.getcwd()
scale=5
peak_scale=8
input = pd.read_csv('./'+ref_array+'/input.csv',header=None)
a=input.to_dict('series')
keys = a[0][:]
values = a[1][:]
res = {}
for i in range(len(keys)):
        res[keys[i]] = values[i]
        #print(keys[i],values[i])
#################################################################
# bp info
## BP parameters from the input file
try:
    bp_l = sys.argv[2]
    bp_u = sys.argv[3]
    print('bp_l and bp_u is,',(bp_l,bp_u))
except:
    bp_l                = float(res['bp_l']) #Hz
    bp_u                = float(res['bp_u'])   #Hz
#bp_l                = float(res['bp_l']) #Hz
#bp_u                = float(res['bp_u'])   #Hz
smooth_time_window  = int(res['smooth_time_window'])   #seconds
smooth_space_window = int(res['smooth_space_window'])
stack_start         = int(res['stack_start'])   #in seconds
stack_end           = int(res['stack_end'])  #in seconds
STF_start           = int(res['STF_start'])
STF_end             = int(res['STF_end'])
sps                 = int(res['sps'])  #samples per seconds
threshold_correlation=float(res['threshold_correlation'])
SNR=float(res['SNR'])
smooth_time_window=10
smooth_space_window=1
STF_start=0
STF_end=40
#stack_start=30
stack_end=70
#bp_l=0.8
#bp_u=5
##########################################################################
# Event info
Event=res['Event']
event_lat=float(res['event_lat'])
event_long=float(res['event_long'])
event_depth=float(res['event_depth'])
#Array_name=res['Array_name']
#Exp_name=res['Exp_name']
origin_time=obspy.UTCDateTime(int(res['origin_year']),int(res['origin_month']),
             int(res['origin_day']),int(res['origin_hour']),int(res['origin_minute']),float(res['origin_seconds']))
print(origin_time)
Focal_mech = dict(strike=float(res['event_strike']), dip=float(res['event_dip']), rake=float(res['event_rake'])
                 , magnitude=float(res['event_magnitude']))
model               = TauPyModel(model=str(res['model']))
sps                 = int(res['sps'])  #samples per seconds
source_grid_size    = float(res['source_grid_size']) #degrees
source_grid_extend  = float(res['source_grid_extend'])   #degrees
source_depth_size   = float(res['source_depth_size']) #km
source_depth_extend = float(res['source_grid_extend']) #km
#stream_for_bp=obspy.read('./Turky_7.6_all/stream.mseed')
slong,slat          = bp_lib.make_source_grid(event_long,event_lat,source_grid_extend,source_grid_size)
##############################
# Finding index of the hypocentral grid
dist=[]
for i in range(len(slat)):
        dist.append(((slat[i]-event_lat)**2 + (slong[i]-event_long)**2 )**0.2);
hypocentre_index=np.argmin(dist)

### Load array beams
beam_EU=np.loadtxt('beam_'+str(bp_l)+'_'+str(bp_u)+'_EU.dat')
beam_AU_=np.loadtxt('beam_'+str(bp_l)+'_'+str(bp_u)+'_AU.dat')
EU_ref= beam_EU[hypocentre_index,:]
AU_ref= beam_AU_[hypocentre_index,:]
max_EU_ref = np.max(EU_ref)
max_AU_ref = np.max(AU_ref)

cc = obspy.signal.cross_correlation.correlate(EU_ref,AU_ref, 10)
shift, corr = obspy.signal.cross_correlation.xcorr_max(cc)
shift = int(shift/sps)
print('Shift=',shift)
print('Corr=',corr)

beam_AU=beam_AU_[:,shift:shift+stack_end*sps]
beam_sum =  (beam_EU[:,0:stack_end*sps] + beam_AU*corr)
#beam_sum = beam_sum/np.max(beam_sum)
#beam_sum =  (beam_EU[:,0:stack_end*sps]*(max_EU_ref/max_AU_ref) + beam_AU*(max_EU_ref/max_AU_ref) )
beam_sum = beam_sum/np.max(beam_sum)

m,n=np.shape(beam_sum)
beam_sum_averaged_time=np.zeros((m,stack_end-stack_start))
beam_AU_averaged_time=np.zeros((m,stack_end-stack_start))
beam_EU_averaged_time=np.zeros((m,stack_end-stack_start))

for i in range(stack_end-stack_start):
    beam_sum_averaged_time[:,i] = bp_lib.moving_average_time_beam(beam_sum[:,i*sps:(i+smooth_time_window-1)*sps]**2)
    beam_AU_averaged_time[:,i] = bp_lib.moving_average_time_beam(beam_AU[:,i*sps:(i+smooth_time_window-1)*sps]**2)
    beam_EU_averaged_time[:,i] = bp_lib.moving_average_time_beam(beam_EU[:,i*sps:(i+smooth_time_window-1)*sps]**2)

#beam_sum_averaged_time= beam_sum_averaged_time**2 #beam_sum_averaged_time/np.max(beam_sum_averaged_time)
#beam_AU_averaged_time= beam_AU_averaged_time**2 #beam_AU_averaged_time/np.max(beam_AU_averaged_time)
#beam_EU_averaged_time= beam_EU_averaged_time**2 #beam_EU_averaged_time/np.max(beam_EU_averaged_time)


#beam_sum_averaged_time= beam_sum_averaged_time**2 #beam_sum_averaged_time/np.max(beam_sum_averaged_time)
#beam_AU_averaged_time= beam_AU_averaged_time**2 #beam_AU_averaged_time/np.max(beam_AU_averaged_time)
#beam_EU_averaged_time= beam_EU_averaged_time**2 #beam_EU_averaged_time/np.max(beam_EU_averaged_time)


#################################
## calculating STF
stf_averaged_combined  = np.mean(beam_sum_averaged_time,axis=0)
stf_averaged_AU  = np.mean(beam_AU_averaged_time,axis=0)
stf_averaged_EU  = np.mean(beam_EU_averaged_time,axis=0)

#Normalizing 
stf_averaged_combined  = stf_averaged_combined/np.max(stf_averaged_combined)
stf_averaged_AU  = stf_averaged_AU/np.max(stf_averaged_AU)
stf_averaged_EU  = stf_averaged_EU/np.max(stf_averaged_EU)

stf_time      = range(len(stf_averaged_combined)) #np.arange(0, len(stf_averaged), 1/sps)
STF_array_combined     = np.copy(stf_time)
STF_array_combined     = np.column_stack((STF_array_combined,stf_averaged_combined)) 

STF_array_AU     = np.copy(stf_time)
STF_array_AU     = np.column_stack((STF_array_AU,stf_averaged_AU)) 

STF_array_EU     = np.copy(stf_time)
STF_array_EU     = np.column_stack((STF_array_EU,stf_averaged_EU)) 


#saving STF 
file_save    = 'STF_beam_'+str(bp_l)+'_'+str(bp_u)+'_combined.dat'
np.savetxt(file_save,STF_array_combined)

file_save    = 'STF_beam_'+str(bp_l)+'_'+str(bp_u)+'_AU.dat'
np.savetxt(file_save,STF_array_AU)

file_save    = 'STF_beam_'+str(bp_l)+'_'+str(bp_u)+'_EU.dat'
np.savetxt(file_save,STF_array_EU)


# Cumulative energy
temp     =np.sum(beam_sum_averaged_time[:,stack_start:stack_end],axis=1)
np.size(temp)
m,n=np.shape(beam_sum_averaged_time)
cumulative_energy=np.zeros((m,3))
cumulative_energy[:,2]=temp/np.max(temp)
cumulative_energy[:,0]=slong
cumulative_energy[:,1]=slat
file_save='cumulative_energy_'+str(bp_l)+'_'+str(bp_u)+'_combined.dat'
np.savetxt(file_save,cumulative_energy)

m,n=np.shape(beam_sum_averaged_time)
peak_energy=np.zeros((n,4))
for i in range(n):
    ind              = np.argmax(beam_sum_averaged_time[:,i])
    peak_energy[i,0] = i
    peak_energy[i,1] = slong[ind]
    peak_energy[i,2] = slat[ind]
    peak_energy[i,3] = beam_sum_averaged_time[ind,i]

file_save='Peak_energy_'+str(bp_l)+'_'+str(bp_u)+'_combined.dat'
np.savetxt(file_save,peak_energy)

colors=range(0,n,1)
fig, ax  = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(10, 6))
tri      = Triangulation(cumulative_energy[:,0],cumulative_energy[:,1])
energy   = ax[0].tricontourf(tri, cumulative_energy[:,2],cmap='copper',levels=np.arange(0, 1,0.1))
event    = ax[0].plot(event_long,event_lat,'*',markersize=20)
cmap     = plt.get_cmap('gnuplot',20)
#cmap.set_under('gray')
peak     = ax[0].scatter(peak_energy[:,1],peak_energy[:,2],c=peak_energy[:,0],
        s=peak_energy[:,3],cmap=cmap,vmin=0,vmax=peak_energy[:,0].max())
fig.colorbar(energy,ax=ax[0],label='Cumulative energy',orientation='horizontal')
fig.colorbar(peak,ax=ax[0],label='Peak energy',orientation='vertical')

stf1     = ax[1].plot(STF_array_combined[:,0],STF_array_combined[:,1],label='Combined')
stf2     = ax[1].plot(STF_array_AU[:,0],STF_array_AU[:,1],label='AU')
stf3     = ax[1].plot(STF_array_EU[:,0],STF_array_EU[:,1],label='EU')
ax[1].legend()
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Normalized Amplitude')

fig.savefig('BP_Peak_energy_'+str(bp_l)+'_'+str(bp_u)+'_combined.png')
