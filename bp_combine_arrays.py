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



###########################
# Event info
###########################
origin_time      = obspy.UTCDateTime(2016, 1, 3, 23, 5, 22)
event_lat        = 24.80360
event_long       = 93.65050
event_depth      = 55.0 # km
###########################
# data info
###########################
##########################
# BP parameters
##########################
sps                 = 20  #samples per seconds
bp_l                = 0.2 #Hz
bp_u                = 5   #Hz
stack_start         = 0   #in seconds
stack_end           = 50  #in seconds
smooth_time_window  = 2   #seconds
source_grid_size    = 0.1 #degrees
source_grid_extend  = 2   #degrees
###############################
# Making potential sources grid
###############################
slong,slat=bp_lib.make_source_grid(event_long,event_lat,source_grid_extend,source_grid_size)
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
