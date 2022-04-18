##########################################################################
# ADD SOME GENERAL INFO and LICENSE -> @ajay6763
##########################################################################
from __future__ import division
import obspy
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.cross_correlation import xcorr_pick_correction # for cross-correlation
from obspy.signal.trigger import recursive_sta_lta_py
from scipy import signal

from bisect import bisect_left
from copy import copy
import warnings
from obspy.signal.invsim import cosine_taper
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib.transforms as mtransforms



def crosscorr(t1,t2,P_cut,window,sps):
    '''
    
    '''
    st  = int((P_cut-window)*sps)
    end = int((P_cut+window)*sps)
    #print('Start:', (st), 'End :',  (end))
    corr,lags=xcorr(t1[st:end],t2[st:end]);
    
    ## find location of maximum correlation
    ind=np.where(corr==max(corr))
    temp=ind[0].item()
    #print(corr[temp])
    shift=lags[temp]/sps;
    if (corr[temp] < 0):
        sign=-1;
    else:
        sign=1;
    return corr[temp],shift,sign

def xcorr_pick_correction(pick1, trace1, pick2, trace2, t_before, t_after,
                          cc_maxlag, filter=None, filter_options={}):
    """
    Calculate the correction for the differential pick time determined by cross
    correlation of the waveforms in narrow windows around the pick times.
    For details on the fitting procedure refer to [Deichmann1992]_.

    The parameters depend on the epicentral distance and magnitude range. For
    small local earthquakes (Ml ~0-2, distance ~3-10 km) with consistent manual
    picks the following can be tried::

        t_before=0.05, t_after=0.2, cc_maxlag=0.10,
        filter="bandpass", filter_options={'freqmin': 1, 'freqmax': 20}

    The appropriate parameter sets can and should be determined/verified
    visually using the option `plot=True` on a representative set of picks.

    To get the corrected differential pick time calculate: ``((pick2 +
    pick2_corr) - pick1)``. To get a corrected differential travel time using
    origin times for both events calculate: ``((pick2 + pick2_corr - ot2) -
    (pick1 - ot1))``

    :type pick1: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param pick1: Time of pick for `trace1`.
    :type trace1: :class:`~obspy.core.trace.Trace`
    :param trace1: Waveform data for `pick1`. Add some time at front/back.
            The appropriate part of the trace is used automatically.
    :type pick2: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param pick2: Time of pick for `trace2`.
    :type trace2: :class:`~obspy.core.trace.Trace`
    :param trace2: Waveform data for `pick2`. Add some time at front/back.
            The appropriate part of the trace is used automatically.
    :type t_before: float
    :param t_before: Time to start cross correlation window before pick times
            in seconds.
    :type t_after: float
    :param t_after: Time to end cross correlation window after pick times in
            seconds.
    :type cc_maxlag: float
    :param cc_maxlag: Maximum lag/shift time tested during cross correlation in
        seconds.
    :type filter: str
    :param filter: `None` for no filtering or name of filter type
            as passed on to :meth:`~obspy.core.trace.Trace.filter` if filter
            should be used. To avoid artifacts in filtering provide
            sufficiently long time series for `trace1` and `trace2`.
    :type filter_options: dict
    :param filter_options: Filter options that get passed on to
            :meth:`~obspy.core.trace.Trace.filter` if filtering is used.
    :type plot: bool
    :param plot: If `True`, a plot window illustrating the alignment of the two
        traces at best cross correlation will be shown. This can and should be
        used to verify the used parameters before running automatedly on large
        data sets.
    :type filename: str
    :param filename: If plot option is selected, specifying a filename here
            (e.g. 'myplot.pdf' or 'myplot.png') will output the plot to a file
            instead of opening a plot window.
    :rtype: (float, float)
    :returns: Correction time `pick2_corr` for `pick2` pick time as a float and
            corresponding correlation coefficient.
    """
    # perform some checks on the traces
    if trace1.stats.sampling_rate != trace2.stats.sampling_rate:
        msg = "Sampling rates do not match: %s != %s" % \
            (trace1.stats.sampling_rate, trace2.stats.sampling_rate)
        raise Exception(msg)
    #if trace1.id != trace2.id:
    #    msg = "Trace ids do not match: %s != %s" % (trace1.id, trace2.id)
    #    warnings.warn(msg)
    samp_rate = trace1.stats.sampling_rate
    # don't modify existing traces with filters
    if filter:
        trace1 = trace1.copy()
        trace2 = trace2.copy()
    # check data, apply filter and take correct slice of traces
    slices = []
    for _i, (t, tr) in enumerate(((pick1, trace1), (pick2, trace2))):
        start = t - t_before - (cc_maxlag / 2.0)
        end = t + t_after + (cc_maxlag / 2.0)
        duration = end - start
        # check if necessary time spans are present in data
        if tr.stats.starttime > start:
            msg = "Trace %s starts too late." % _i
            raise Exception(msg)
        if tr.stats.endtime < end:
            msg = "Trace %s ends too early." % _i
            raise Exception(msg)
        if filter and start - tr.stats.starttime < duration:
            msg = "Artifacts from signal processing possible. Trace " + \
                  "%s should have more additional data at the start." % _i
            warnings.warn(msg)
        if filter and tr.stats.endtime - end < duration:
            msg = "Artifacts from signal processing possible. Trace " + \
                  "%s should have more additional data at the end." % _i
            warnings.warn(msg)
        # apply signal processing and take correct slice of data
        if filter:
            tr.data = tr.data.astype(np.float64)
            tr.detrend(type='demean')
            tr.data *= cosine_taper(len(tr), 0.1)
            tr.filter(type=filter, **filter_options)
        slices.append(tr.slice(start, end))
    # cross correlate
    shift_len = int(cc_maxlag * samp_rate)
    cc = obspy.signal.cross_correlation.correlate(slices[0].data, slices[1].data, shift_len, method='direct')
    cc = abs(cc)
    _cc_shift, cc_max = obspy.signal.cross_correlation.xcorr_max(cc)
    cc_curvature = np.concatenate((np.zeros(1), np.diff(cc, 2), np.zeros(1)))
    cc_convex = np.ma.masked_where(np.sign(cc_curvature) >= 0, cc)
    cc_concave = np.ma.masked_where(np.sign(cc_curvature) < 0, cc)
    # check results of cross correlation
    #if cc_max < 0:
    #    msg = "Absolute maximum is negative: %.3f. " % cc_max + \
    #          "Using positive maximum: %.3f" % max(cc)
    #    warnings.warn(msg)
    #    cc_max = max(cc)
    #if cc_max < 0.8:
    #    msg = "Maximum of cross correlation lower than 0.8: %s" % cc_max
    #    warnings.warn(msg)
    # make array with time shifts in seconds corresponding to cc function
    cc_t = np.linspace(-cc_maxlag, cc_maxlag, shift_len * 2 + 1)
    # take the subportion of the cross correlation around the maximum that is
    # convex and fit a parabola.
    # use vertex as subsample resolution best cc fit.
    peak_index = cc.argmax()
    first_sample = peak_index
    # XXX this could be improved..
    while first_sample > 0 and cc_curvature[first_sample - 1] <= 0:
        first_sample -= 1
    last_sample = peak_index
    while last_sample < len(cc) - 1 and cc_curvature[last_sample + 1] <= 0:
        last_sample += 1
    if first_sample == 0 or last_sample == len(cc) - 1:
        msg = "Fitting at maximum lag. Maximum lag time should be increased."
        warnings.warn(msg)
    # work on subarrays
    num_samples = last_sample - first_sample + 1
    if num_samples < 3:
        msg = "Less than 3 samples selected for fit to cross " + \
              "correlation: %s" % num_samples
        raise Exception(msg)
    if num_samples < 5:
        msg = "Less than 5 samples selected for fit to cross " + \
              "correlation: %s" % num_samples
        warnings.warn(msg)
    # quadratic fit for small subwindow
    coeffs, residual = np.polyfit(
        cc_t[first_sample:last_sample + 1],
        cc[first_sample:last_sample + 1], deg=2, full=True)[:2]
    # check results of fit
    if coeffs[0] >= 0:
        msg = "Fitted parabola opens upwards!"
        warnings.warn(msg)
    if residual > 0.1:
        msg = "Residual in quadratic fit to cross correlation maximum " + \
              "larger than 0.1: %s" % residual
        warnings.warn(msg)
    # X coordinate of vertex of parabola gives time shift to correct
    # differential pick time. Y coordinate gives maximum correlation
    # coefficient.
    dt = -coeffs[1] / 2.0 / coeffs[0]
    coeff = (4 * coeffs[0] * coeffs[2] - coeffs[1] ** 2) / (4 * coeffs[0])
    # this is the shift to apply on the time axis of `trace2` to align the
    # traces. Actually we do not want to shift the trace to align it but we
    # want to correct the time of `pick2` so that the traces align without
    # shifting. This is the negative of the cross correlation shift.
    dt = -dt
    pick2_corr = dt
    return (pick2_corr, coeff)

def make_source_grid(event_long,event_lat,source_grid_extend,source_grid_size):
    '''
    This function makes potential source grid around the epicentre in a area 
    defined by a constant source_grid_extend discretized at a constant 
    source_grid_size
    Retunrs   slat ,slong 

    '''
    x=np.arange(event_long-source_grid_extend,event_long+source_grid_extend,source_grid_size)
    y=np.arange(event_lat-source_grid_extend,event_lat+source_grid_extend,source_grid_size)
    slat = []
    slong = []
    for i in range(np.size(x)):
        for j in range(np.size(y)):
            slong.append(x[i])
            slat.append(y[j])
    return slong,slat

def check_sps(stream):
    '''
    This function checks if all the waveform data has 20 SPS. At the moment it can detect
    all the possible values and can decimate to 20 SPS. 
    Sometimes waveforms have a SPS which not interger multiple of 20 SPS, I simply reject them.
    Yes, you can decimate and interpolate these waveforms back 20 SPS but I choose not to play with
    the signal and try to make them as original as possible without the interpolation that might
    introduce "ärtifacts".
    @ajay6763: MAKE THIS A ROBUST FUNCTION. 
    '''
    # make a copy of the data and leave the original
    stream_work=stream.copy()
    for t in stream_work:
        if (t.stats.sampling_rate==20. or t.stats.sampling_rate==40. or t.stats.sampling_rate==80.
            or t.stats.sampling_rate==100. or t.stats.sampling_rate==120. or t.stats.sampling_rate==200.):
            pass
        #t.decimate(2)
        #t.write
        else:
            stream_work.remove(t)
    
    print('Total no of traces before decimation criteria:', len(stream))
    print('Total no of traces after decimation criteria:', len(stream_work))
    for t in stream_work:
        if (t.stats.sampling_rate==20.):
             pass
        elif (t.stats.sampling_rate==40.):
             t.decimate(2)
        elif (t.stats.sampling_rate==50.):
             t.resample(20.0)
        elif (t.stats.sampling_rate==80.):
             t.decimate(4)
        elif (t.stats.sampling_rate==100.):
             t.decimate(5)
        elif (t.stats.sampling_rate==120.):
             t.decimate(6)
        elif (t.stats.sampling_rate==200.):
             t.decimate(10)
        else:
            print("There are some traces that cannot be decimated 20 SPS. Please check the SPS of your data")

def check_distance(stream,min_distance,max_distance):
    '''
    This functions checks for distance (degrees) and only selecet waveforms
    that are within a specified distance rage (i.e. to avoid core phases)
    '''
    # make a copy of the data and leave the original
    stream_work=stream.copy()
    print('Total no of traces before :', len(stream))
    for t in stream_work:
        if (t.stats.Dist > min_distance and t.stats.Dist < max_distance):
            pass
        else:
            stream_work.remove(t)
    print('Total no of traces after :', len(stream_work))
    return stream_work    

def cut_window(trace,T,Start,End):
    '''
    '''
    ## find index corresponding to the calculated travel time
    arrival_index = int((T-trace.stats.starttime)*trace.stats.sampling_rate)
    start_index   = int((T+Start-trace.stats.starttime)*trace.stats.sampling_rate)
    end_index     = int((T+End-trace.stats.starttime)*trace.stats.sampling_rate) 
    data          = trace.data
    data          = data/np.max(data)
    cut           = data[start_index:end_index]
    width         = end_index-start_index
    # Finding sign of the wave at the arrival time
    sign  = 1
    #if (data[arrival_index] < 0):
    #    sign = -1
    #else:
    #    pass
            
    return cut,width,sign

def xcorr(x,y):
    """
    Perform Cross-Correlation on x and y
    x    : 1st signal
    y    : 2nd signal

    returns
    lags : lags of correlation
    corr : coefficients of correlation
    """
    corr = signal.correlate(x, y, mode="full")
    lags = signal.correlation_lags(len(x), len(y), mode="full")
    return corr,lags
def crosscorr(t1,t2,P_cut,window,sps):
    '''
    
    '''
    st  = int((P_cut-window)*sps)
    end = int((P_cut+window)*sps)
    #print('Start:', (st), 'End :',  (end))
    corr,lags=xcorr(t1[st:end],t2[st:end]);
    
    ## find location of maximum correlation
    ind=np.where(corr==max(corr))
    temp=ind[0].item()
    #print(corr[temp])
    shift=lags[temp]/sps;
    if (corr[temp] < 0):
        sign=-1;
    else:
        sign=1;
    return corr[temp],shift,sign


def moving_average_time(data, w):
    return np.convolve(data, np.ones(w), 'same') / w

def plot_array(stream,event_long,event_lat,Array_name,Ref_station_index):
    '''
    '''
    sta_lat=[]
    sta_long=[]
    for tr in stream:
        plt.plot(tr.stats.station_longitude,tr.stats.station_latitude,'^')
        sta_lat.append(tr.stats.station_latitude)
        sta_long.append(tr.stats.station_longitude)

    plt.plot(event_long,event_lat,'*',label='Earthquake')
    plt.plot(sta_long[Ref_station_index],sta_lat[Ref_station_index],'o',color='b',label='Reference')
    plt.legend()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    station_save=np.copy(sta_long)
    station_save=np.column_stack((station_save,sta_lat))
    np.savetxt(str(Array_name)+'_station_list.dat',station_save)
    plt.savefig(str(Array_name)+'_BP_stations.png')
def plot_results(beam_plot,stf,event_long,event_lat,Array_name,slong,slat,stack_start,stack_end):
    '''
    '''
    fig, ax = plt.subplots(4, 4, sharex=False, sharey=False,figsize=(16, 22))

    tri = Triangulation(slong[:],slat[:])
    time = [0,4,8,12,
            16,20,24,28,
            32,36,40,44,
           48,52,56,60]
    for i in range(4):
        for j in range(4):
            energy = ax[i][j].tricontourf(tri, beam_plot[:,i*3 + j],cmap='hot',levels=np.arange(0, 1,0.1))
            eq     = ax[i][j].plot(event_long,event_lat,'*',markersize=14)
            ax[i][j].set_title(str(time[i*4 + j]) +' seconds')
            #ax[i][j].set_xlim((event_long-0.5,event_long+0.5))
            #ax[i][j].set_ylim((event_lat-0.5,event_lat+0.5))
            fig.colorbar(energy, ax=ax[i][j], label='Energy', orientation='horizontal')
    fig.savefig(str(Array_name)+'_BP_time_evolution.png', dpi=fig.dpi)
    fig2, ax2 = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(10, 6))
    # Cumulative energy
    temp     =np.sum(beam_plot[:,stack_start:stack_end],axis=1)
    np.size(temp)
    cumulative_energy=temp/np.max(temp)
    tri    = Triangulation(slong[:],slat[:])
    energy_cum = ax2[0].tricontourf(tri, cumulative_energy,cmap='hot',levels=np.arange(0, 1,0.1))
    eq     = ax2[0].plot(event_long,event_lat,'*',markersize=14)
    ax2[0].set_title('Cumulative energy')
    #ax[i][j].set_xlim((event_long-0.5,event_long+0.5))
    #ax[i][j].set_ylim((event_lat-0.5,event_lat+0.5))
    fig.colorbar(energy, ax=ax2[0], label='Cumulative Energy', orientation='horizontal')
    s       = ax2[1].plot(stf[:,0],stf[:,1],'*',markersize=2)
    ax2[1].set_xlabel('Time (s)')
    ax2[1].set_ylabel('Amplitude ')
    ax2[1].set_title('STF')
    fig2.savefig(str(Array_name)+'_BP_cumulative_STF.png', dpi=fig.dpi)
    
    
def moving_average_time_beam(data):
    return np.sum(data[:,:],axis=1) 

def moving_average_space(data):
    return np.sum(data[:,:],axis=0) 
        