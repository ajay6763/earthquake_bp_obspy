##########################################################################
# ADD SOME GENERAL INFO and LICENSE -> @ajay6763
##########################################################################
from __future__ import division
import obspy
import sys,os,time

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
import math
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib.transforms as mtransforms
from mpl_toolkits.basemap import Basemap
from obspy.imaging.beachball import beach
import numpy as np
########### 
def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.3+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}".format(prefix, "#"*x, "."*(size-x), j, count), 
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)
def save_stream_info(stream_for_bp):
    sta = []
    sta_lat = []
    sta_long = []
    dist = []
    azimuth = []
    corr = []
    shift = []
    sign = []
    P_arrival = []
    for tr in stream_for_bp:
        sta.append(tr.stats.station)
        sta_long.append(tr.stats.station_longitude)
        sta_lat.append(tr.stats.station_latitude)
        dist.append(tr.stats.Dist)
        azimuth.append(tr.stats.Azimuth)
        P_arrival.append(tr.stats.P_arrival)
        corr.append(tr.stats.Corr_coeff)
        shift.append(tr.stats.Corr_shift)
        sign.append(tr.stats.Corr_sign)
    to_save = np.zeros_like(sta)
    to_save = np.column_stack((to_save,sta))
    to_save = np.column_stack((to_save,sta_long))
    to_save = np.column_stack((to_save,sta_lat))
    to_save = np.column_stack((to_save,dist))
    to_save = np.column_stack((to_save,azimuth))
    to_save = np.column_stack((to_save,P_arrival))
    to_save = np.column_stack((to_save,corr))
    to_save = np.column_stack((to_save,shift))
    to_save = np.column_stack((to_save,sign))
    #np.save('array_bp_info',to_save,allow_pickle=True)
    return to_save
def nth_root_stacking(arr, n):
    """
    This function performs nth root stacking on a NumPy array
    
    Parameters:
    arr (np.ndarray): NumPy array
    n (int): nth root value
    
    Returns:
    np.ndarray: nth root stacked array
    """
    
    # Ensure that the input array is a NumPy array
    arr = np.array(arr)
    
    # Perform nth root stacking
    arr_stacked = np.vstack(np.power(arr, 1.0/n))
    
    return arr_stacked
'''
def station_info(stations,stream)
'''
def polarity(stream,window):
    sign=1
    for tr in stream:
        t_corr = tr.stats['P_arrival'] + tr.stats.Corr_shift
        cut = cut_window(tr, t_corr, -1*window, window)[0]
        #cut=cut/np.max(cut)
        mean=np.mean(cut)
        if mean<0:
            tr.stats['Corr_sign']=-1
        else:
            tr.stats['Corr_sign']=-1
    return stream
def event_plot(event_lat,event_long,sta_lat,sta_long,name):
    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False,figsize=(7,5))

    #ind = np.where( (np.asarray(sta_azimuth[:])>=azimuth_min) & (np.asarray(sta_azimuth[:])<=azimuth_max) )

    #map = Basemap(ax=ax,projection='merc',llcrnrlon=np.min(rheology[:,0]),llcrnrlat=np.min(rheology[:,1]),urcrnrlon=np.max(rheology[:,0]),urcrnrlat=np.max(rheology[:,1]),resolution='i',fix_aspect=2
    #         )
    map = Basemap(ax=ax,projection='aeqd',lon_0=event_long,lat_0=event_lat)

    #map = Basemap(ax=ax,projection='npstere',lon_0=37)
    #event_plot= map.scatter(event_long,event_lat,latlon=True,facecolor='none',edgecolors='red',marker='o',s=2,linewidths=0.2)
    event_plot= map.scatter(event_long,event_lat,latlon=True,Truefacecolor='black',marker='*',label='Event')
    event_plot= map.scatter(sta_long[:],sta_lat[:],latlon=True,facecolor='pink',marker='^',label='Selected')


    # add an axis for the z colorbar
    #cbar_ax = fig.add_axes()
    # draw the colorbar
    #cb = fig.colorbar(strength, cax=cbar_ax, label='strength',extend='both',pad=0.01)
    map.drawcoastlines()
    #map.shadedrelief()

    #map.drawparallels(np.arange(-90,90,10),labels=[1,0,0,0])
    #map.drawmeridians(np.arange(-180,180,10),labels=[1,1,0,1], rotation=0)
    #map.c(rheology[ind[0][:],0],rheology[ind[0][:],1],latlon=True)
    # fill continents 'coral' (with zorder=0), color wet areas 'aqua'
    #map.drawmapboundary(fill_color='aqua')
    #map.fillcontinents(color='coral',lake_color='aqua')
    #plt.title('Day/Night Map for %s (UTC)')
    #map.colorbar(label='sdfdsf')
    #fig.colorbar(map,ax=map,orientation='horizontal',label='sdfdsf')
    #ax.set_title('')
    ax.set_xlabel('Longitude',labelpad=30)
    ax.set_ylabel('Latitude', labelpad=40)
    plt.legend()
    #fig.suptitle(str(code)+' age')
    plt.show()
    plt.savefig(name+'.png',dpi=450)

def data_plot(stream,event_lat,event_long,outdir,outname):
    fig, ax = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(10,3))
    #map =  Basemap(projection='cyl', lon_0=event_long,lat_0=event_lat,
    #        resolution='c',ax=ax[0])
    width = 28e6
    #map = Basemap(width=width,height=width,projection='aeqd',lon_0=event_long,lat_0=event_lat,resolution='c',ax=ax[1])
    map = Basemap(ax=ax[0],width=width,height=width,projection='aeqd',lon_0=event_long,lat_0=event_lat,resolution='c')

    for tr in stream:
        map.scatter(tr.stats.station_longitude,tr.stats.station_latitude,latlon=True,facecolor='blue',marker='^')
        time = np.arange(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.delta)
        #tr.plot(starttime=t.stats.P_arrival-30,endtime=t.stats.P_arrival+60,type='relative')
        #tr.plot(type='relative')
        tr.normalize
        ax[1].plot(time,tr.data /np.max(tr.data),color='gray')
        #ax[1].plot(time,tr.data,color='gray')
    map.scatter(event_long,event_lat,latlon=True,facecolor='red',marker='*')
    map.drawcoastlines(linewidth=0.1)
    x, y = map(event_long, event_lat)
    #focmecs = Focal_mech
    #ax = plt.gca()
    #b = beach(focmecs, xy=(x, y), width=10, linewidth=1, alpha=0.85)
    #b.set_zorder(10000000)
    #ax.add_collection(b)
    ## plot traces
    plt.show()
    fig.savefig(outdir+'/'+outname)



def get_ref_station(stream):
    """
    """
    x=[]
    y=[]
    for tr in stream:
        x.append(tr.stats.station_longitude)
        y.append(tr.stats.station_latitude)
    n = len(x)
    x_sum = np.sum(x)
    y_sum = np.sum(y)
    x_centroid = x_sum / n
    y_centroid = y_sum / n

    x_ref = None
    y_ref = None
    dist=np.array(np.sqrt((x[:]-x_centroid)**2+(y[:]-y_centroid)**2));
    index=np.argmin(dist);
    return index


def get_ref_station_pev(stream,type,out_option):
    sta_dist=[]
    sta_azimuth=[]
    for tr in stream:
        sta_dist.append(tr.stats.Dist)
        sta_azimuth.append(tr.stats.Azimuth)
    if type=='mean':
        mean_dist = np.mean(sta_dist)
        dist=np.array((mean_dist-sta_dist)**2);
        index_dist=dist.argmin();

        mean_azimuth = np.mean(sta_azimuth)
        dist=np.array((mean_azimuth-sta_azimuth)**2);
        index_azimuth=dist.argmin();
    else:
        median_dist = np.median(sta_dist)
        dist=np.array((median_dist-sta_dist)**2);
        index_dist=dist.argmin();
        median_azimuth = np.median(sta_azimuth)
        dist=np.array((median_azimuth-sta_azimuth)**2);
        index_azimuth=dist.argmin();

    if out_option=='dist':
        Ref_station_index=index_dist
    else:
        Ref_station_index=index_azimuth
    return Ref_station_index


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

def crosscorr_prev(t1_trace,t2_trace,window):
    '''

    '''
    sps=int(t1_trace.stats['sampling_rate'])
    #cc=obspy.signal.cross_correlation.correlate(t1_trace[ref_st:ref_end],t2_trace[st:end],demean=True,normalize='naive',method='auto',shift=window*sps)
    cc=obspy.signal.cross_correlation.correlate(t1_trace,t2_trace,demean=True,normalize='naive',method='auto',shift=window*sps)
    shift, value = obspy.signal.cross_correlation.xcorr_max(cc)
    if (value < 0):
            sign=-1;
    else:
            sign=1;

    return abs(value),shift/sps,sign
def crosscorr_stream_prev(stream,ref_trace,window):
    '''
    '''
    for tr in stream:
        corr,shift,sign = crosscorr_prev(ref_trace,tr,window)
        tr.stats['Corr_coeff'] = corr
        tr.stats['Corr_shift']  = shift
        tr.stats['Corr_sign']  = sign
        #tr.stats['Corr_coeff'] = 1
        #tr.stats['Corr_shift']  = 0
        #tr.stats['Corr_sign']  = 1
        #except:
        #    stream.remove(tr)
    return stream
def crosscorr(t1_trace,t2_trace,window):
    '''

    '''
    sps=int(t1_trace.stats['sampling_rate'])
    ref_st  = int(t1_trace.stats['P_arrival']-t1_trace.stats['origin_time']-window)*sps
    ref_end = int(t1_trace.stats['P_arrival']-t1_trace.stats['origin_time']+window)*sps
    st  = int(t2_trace.stats['P_arrival']-t2_trace.stats['origin_time']-window)*sps
    end = int(t2_trace.stats['P_arrival']-t2_trace.stats['origin_time']+window)*sps
    ref_st  = int(t1_trace.stats['P_arrival']-t1_trace.stats['origin_time']-window)*sps
    ref_end = int(t1_trace.stats['P_arrival']-t1_trace.stats['origin_time']+window)*sps
    st  = int(t2_trace.stats['P_arrival']-t2_trace.stats['origin_time']-window)*sps
    end = int(t2_trace.stats['P_arrival']-t2_trace.stats['origin_time']+window)*sps
    message='ok'
    try:
        #print('ok!')
        cc=obspy.signal.cross_correlation.correlate(t1_trace[ref_st:ref_end],t2_trace[st:end],demean=True,normalize='naive',method='auto',shift=window*sps)
        #cc=obspy.signal.cross_correlation.correlate(t1_trace,t2_trace,demean=True,normalize='naive',method='auto',shift=window*sps)
        message='ok'
        shift, value = obspy.signal.cross_correlation.xcorr_max(cc)
        #print('Corr, Shift:',(value,shift/sps))
        if (t2_trace[st+shift] < 0):
            sign=-1;
        else:
            sign=1;
    except:
        #print('Not!')
        #cc=obspy.signal.cross_correlation.correlate(t1_trace,t2_trace,demean=True,normalize='naive',method='auto',shift=window*sps)
        message='not ok'
        value=0
        shift=0
        sign=1
    return abs(value),shift/sps,sign,message
def crosscorr_stream(stream,ref_trace,window):
    '''
    '''
    for tr in stream:
        corr,shift,sign,message = crosscorr(ref_trace,tr,window)
        if message=='ok':
            tr.stats['Corr_coeff'] = corr
            tr.stats['Corr_shift']  = shift
            tr.stats['Corr_sign']  = sign
        else:
            stream.remove(tr)
        #except:
        #    stream.remove(tr)
    return stream

def crosscorr_stream_xcorr(stream,ref_trace,window):
    '''
    '''
    for tr in stream:
        '''
        shift, value = xcorr_pick_correction(ref_trace.stats.P_arrival, ref_trace,tr.stats.P_arrival, tr, t_before=5, t_after=10, cc_maxlag=5) #,filter="bandpass",filter_options={'freqmin': bp_l, 'freqmax': bp_u})
        tr.stats['Corr_coeff'] = value
        tr.stats['Corr_shift']  = shift
        tr.stats['Corr_sign']  = 1
        '''
        try:
            shift, value = xcorr_pick_correction(ref_trace.stats.P_arrival, ref_trace,tr.stats.P_arrival, tr,
                t_before=window, t_after=window, cc_maxlag=window/2)#,filter="bandpass",filter_options={'freqmin': 0.2, 'freqmax': 5})
            tr.stats['Corr_coeff'] = value
            tr.stats['Corr_shift']  = shift
            tr.stats['Corr_sign']  = 1
        except:
            print('Could not cross-correlate! Hence remove this waveform.')
            #tr.stats['Corr_coeff'] = corr
            #tr.stats['Corr_shift']  = shift
            #tr.stats['Corr_sign']  = 1
            stream.remove(tr)

    return stream

def snr_calc(tr, noise_window, signal_window):
    """
    """
    t_noise = tr.copy()
    t_signal = tr.copy()
    t_noise.trim(tr.stats['P_arrival']-noise_window,tr.stats['P_arrival'])
    t_signal.trim(tr.stats['P_arrival'],tr.stats['P_arrival']+signal_window)

    signal_amp = np.sqrt(np.mean(np.square(t_signal.data)))
    noise_amp = np.sqrt(np.mean(np.square(t_noise.data)))
    return signal_amp / noise_amp
def snr_check(stream,SNR,t_before,t_after):
    '''
    This function checks if all the waveform data has 20 SPS. At the moment it can detect
    all the possible values and can decimate to 20 SPS.
    Sometimes waveforms have a SPS which not interger multiple of 20 SPS, I simply reject them.
    Yes, you can decimate and interpolate these waveforms back 20 SPS but I choose not to play with
    the signal and try to make them as original as possible without the interpolation that might
    introduce "ärtifacts".
    @ajay6763: MAKE THIS A ROBUST FUNCTION.
    '''
    for t in stream:
        try:
            snr=snr_calc(t,t_before,t_after)
            if (snr >=  SNR):
                pass
            else:
                stream.remove(t)
        except:
            stream.remove(t)
    return stream

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

def make_source_grid_3D(event_long,event_lat,source_grid_extend,source_grid_size,depth_min,depth_max,depth_inc):
    '''
    This function makes potential source grid around the epicentre in a area
    defined by a constant source_grid_extend discretized at a constant
    source_grid_size
    Retunrs   slat ,slong

    '''
    x=np.arange(event_long-source_grid_extend,event_long+source_grid_extend,source_grid_size)
    y=np.arange(event_lat-source_grid_extend,event_lat+source_grid_extend,source_grid_size)
    z=np.arange(depth_min,depth_max,depth_inc)
    slat = []
    slong = []
    sdepth = []
    for i in range(np.size(x)):
        for j in range(np.size(y)):
            for k in range(np.size(z)):
                slong.append(x[i])
                slat.append(y[j])
                sdepth.append(z[k])
    return slong,slat,sdepth
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
    for t in stream:
        if (t.stats.sampling_rate  == 20.):
            pass
        elif (t.stats.sampling_rate  < 15.):
            stream.remove(t)
        else:
            t.resample(20.0)
        #        else:
        #            print("There are some traces that cannot be decimated 20 SPS. Please check the SPS of your data")
    return stream

def check_distance(stream,min_distance,max_distance):
    '''
    This functions checks for distance (degrees) and only selecet waveforms
    that are within a specified distance rage (i.e. to avoid core phases)
    '''
    # make a copy of the data and leave the original
    stream_work=stream.copy()
    #print('Total no of traces before :', len(stream))
    for t in stream_work:
        if (t.stats.Dist >= min_distance and t.stats.Dist <= max_distance):
            pass
        else:
            stream_work.remove(t)
    #print('Total no of traces after :', len(stream_work))
    return stream_work

def check_distance_except(stream,min_distance,max_distance):
    '''
    This functions checks for distance (degrees) and only selecet waveforms
    that are within a specified distance rage (i.e. to avoid core phases)
    '''
    # make a copy of the data and leave the original
    stream_work=stream.copy()
    #print('Total no of traces before :', len(stream))
    for t in stream_work:
        if (t.stats.Dist >= min_distance and t.stats.Dist <= max_distance):
            stream_work.remove(t)
        else:
            pass
    #print('Total no of traces after :', len(stream_work))
    return stream_work

def check_azimuth(stream,min_azimuth,max_azimuth):
    '''
    This functions checks for distance (degrees) and only selecet waveforms
    that are within a specified distance rage (i.e. to avoid core phases)
    '''
    # make a copy of the data and leave the original
    stream_work=stream.copy()
    #print('Total no of traces before :', len(stream))
    for t in stream_work:
        if (t.stats.Azimuth >= min_azimuth and t.stats.Azimuth <= max_azimuth):
            pass
        else:
            stream_work.remove(t)
    #print('Total no of traces after :', len(stream_work))
    return stream_work

def check_baz(stream,min_baz,max_baz):
    '''
    This functions checks for distance (degrees) and only selecet waveforms
    that are within a specified distance rage (i.e. to avoid core phases)
    '''
    # make a copy of the data and leave the original
    stream_work=stream.copy()
    #print('Total no of traces before :', len(stream))
    for t in stream_work:
        if (t.stats.Backazimuth >= min_baz and t.stats.Backazimuth <= max_baz):
            pass
        else:
            stream_work.remove(t)
    #print('Total no of traces after :', len(stream_work))
    return stream_work

def STA_LTA(stream,nsta,nlta,Start_P_cut_time):
    '''
    '''
    for t in stream:
        t.detrend
        t.normalize
        cft         = obspy.signal.trigger.recursive_sta_lta_py(t.data, int(nsta * t.stats.sampling_rate), 
                      int(nlta * t.stats.sampling_rate))
        time        = np.arange(0, t.stats.npts / t.stats.sampling_rate, t.stats.delta)
        ind         = np.argmax(cft)
        t.stats['STA_LTA_pick'] = t.stats.origin_time+time[ind]
        t.stats['STA_LTA_shift'] = t.stats.P_arrival - t.stats.STA_LTA_pick
        #print(t.stats.STA_LTA_pick,t.stats.STA_LTA_shift)

        #if (abs(time[ind]-Start_P_cut_time) > 5 ):
        #    stream.remove(t)
        #    #pass
        #else:
        #    t.stats['STA_LTA_shift'] = time[ind]
         
    return stream
def select_except(stream,min_azimuth,max_azimuth,min_dist,max_dist):
    '''
    This functions checks for distance (degrees) and only selecet waveforms
    that are within a specified distance rage (i.e. to avoid core phases)
    if (sta_azimuth[i]>=azimuth_min and sta_azimuth[i]<=azimuth_max and sta_dist[i] >= dist_min
            and sta_dist[i]<=dist_max):
    '''
    # make a copy of the data and leave the original
    stream_work=stream.copy()
    #print('Total no of traces before :', len(stream))
    for t in stream_work:
        if (t.stats.Azimuth >= min_azimuth and t.stats.Azimuth <= max_azimuth  and t.stats.Dist >= min_dist and t.stats.Dist <= max_dist ):
            print("###################")
        else:
            print("!!!!!!!!!!!!!!!!!!!")
            stream_work.remove(t)
    #print('Total no of traces after :', len(stream_work))
    return stream_work

def check_azimuth_except(stream,min_azimuth,max_azimuth):
    '''
    This functions checks for distance (degrees) and only selecet waveforms
    that are within a specified distance rage (i.e. to avoid core phases)
    '''
    # make a copy of the data and leave the original
    stream_work=stream.copy()
    print('Total no of traces before :', len(stream))
    for t in stream_work:
        if (t.stats.Azimuth >= min_azimuth and t.stats.Azimuth <= max_azimuth):
            stream_work.remove(t)
        else:
            pass
    print('Total no of traces after :', len(stream_work))
    return stream_work

def cut_window(trace,T,Start,End):
    '''
    '''
    ## find index corresponding to the calculated travel time
    arrival_index = int((T-trace.stats.starttime)*trace.stats.sampling_rate)
    #start_index = arrival_index - int(Start*trace.stats.sampling_rate)
    #end_index = arrival_index + int(End*trace.stats.sampling_rate)

    start_index   = int((T-Start-trace.stats.starttime)*trace.stats.sampling_rate)
    end_index     = int((T+End-trace.stats.starttime)*trace.stats.sampling_rate)
    data          = trace.data
    cut           = data[start_index:end_index]
    width         = end_index-start_index
    # Finding sign of the wave at the arrival time
    sign  = 1
    #if (data[arrival_index] < 0):
    #    sign = -1
    #else:
    #    pass

    return cut,width,sign

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
