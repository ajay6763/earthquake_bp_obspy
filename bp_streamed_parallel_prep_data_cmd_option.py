##########################################################################
# ADD SOME GENERAL INFO and LICENSE -> @ajay6763
##########################################################################
from __future__ import division
import sys,os,time, getopt
import obspy
from obspy.taup import TauPyModel
import numpy as np
import csv
import pandas as pd
import bp_lib
from joblib import Parallel, delayed
import obspy.geodetics
import multiprocessing as mp


def process_location(j, slat, slong, stream_for_bp, event_depth, origin_time, model):
    '''
    This function write the source grid  and associated travel times to the stations
    in an array.
    '''
    source_stream_info = []
    
    for t in stream_for_bp:
        distance = obspy.geodetics.locations2degrees(slat[j], slong[j], t.stats.station_latitude, t.stats.station_longitude)
        arrivals = model.get_travel_times(source_depth_in_km=event_depth, distance_in_degree=distance, phase_list=["P"])
        arr = arrivals[0]
        t_travel = arr.time
        t_total = origin_time + t_travel #+ t.stats.Corr_shift
        source_stream_info.append([slat[j], slong[j],t.stats.station,t_total])
    return source_stream_info
def process_beam(j):
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
        cut = tr.data/np.max(np.abs(tr.data)) * tr.stats.Corr_coeff/tr.stats.Station_weight
        stack.append(cut[0:int((stack_start+stack_end)*sps)])
    return np.sum(stack,axis=0)
def main(argv):
    time_start = time.time()
    num_cores = int(1);root_order = int(2);corr_window=float(10);snr_window=float(10);extra_label=str('')
    print('\n###########################################################################################')
    print(' Welcome for help run this script with -h option\n')
    try:
        input_file = sys.argv[-1]
        #print('Input file is:',(input_file))
        Input = pd.read_csv('./'+input_file,header=None)
    except:
        print('You did not provided input file (.csv file). You must run without -h option to use the input_default.csv.')
        message = input('Do you want to continue with default input.csv? (yes/no) :')

        if message=='yes':
            input_file='./input_default.csv'
            Input = pd.read_csv('./'+input_file,header=None)
        else:
            print('\n###########################################################################################')
            print(' A simple run involves following where you specify your input_default.csv in the end.\n\
                    Simple run example: python bp_streamed_parallel_prep_data_cmd_option.py input_default.csv\n\
                    Below are the available options which you can pass as a command line argument:\n\
                    -h : help\n\
                    -p : no of parts to run in parallel (e.g., no of available cores). Default is 1\n\
                    -I : input directory. Default is ./data/\n\
                    -i : input file name in the data_tomo folder. Format x(*) y(*) depth(km) Vs(km/s)\n\
                    -O : output directory. Default is ./output\n\
                    -o : output file name which will be saved in output folder\n\
                    -E : Experiment name. This will be used a\n\
                    -s : Sampling rate i.e. SPS. Default is 20.\n\
                    -a : Comma separated Azimuth range(-180/180) where first is the low and second is max (e.g., 60,90) \n\
                    -d : Comma separated Distnace range in degrees where first is the low and second is max (e.g., 30,90)\n\
                    -B : Comma separated min and max frequency (Hz) for the bandpass filter (0.2/5.0)\n\
                    -C : Threshold cross-correlation coefficient(0-1.0) for waveform selection\n\
                    -S : Signal to noise ratio.Default is 2\n\
                    -G : Source grid extend in degrees. A square grid of this size centered at the hypocenter will be created.\n\
                    -g : Source grid size in degrees.\n\
                    -A : name of the array (e.g., AU,EU etc)\n\
                    \nAll of the input options will be written in the input.csv file in the output directory.')
            print('###########################################################################################\n')               
            sys.exit(2)
    #file='input_EU_7.7.csv'
    a=Input.to_dict('series');keys = a[0][:];values = a[1][:]
    res = {}
    for i in range(len(keys)):
            res[keys[i]] = values[i]
            #print(keys[i],values[i])
    ##########################################################################
    # Event info
    Event=str(res['Event']);event_lat=float(res['event_lat']);event_long=float(res['event_long']);event_depth=float(res['event_depth'])
    Array_name=res['Array_name'];azimuth_min=float(res['azimuth_min']);azimuth_max=float(res['azimuth_max'])
    try:
        backazimuth_min=float(res['backazimuth_min'])
        backazimuth_max=float(res['backazimuth_max'])
    except:
        pass
    dist_min=float(res['dist_min']);dist_max=float(res['dist_max'])
    origin_time=obspy.UTCDateTime(int(res['origin_year']),int(res['origin_month']),
                int(res['origin_day']),int(res['origin_hour']),int(res['origin_minute']),float(res['origin_seconds']))
    #print(origin_time)
    Focal_mech = dict(strike=float(res['event_strike']), dip=float(res['event_dip']), rake=float(res['event_rake'])
                    , magnitude=float(res['event_magnitude']))
    stations = str(res['stations']);waveforms= str(res['waveforms']) 
    ##########################################################################
    # BP parameters
    ##########################################################################
    model               = TauPyModel(model=str(res['model']))
    Start_P_cut_time    = float(res['Start_P_cut_time'])  #before P arrival in seconds
    End_P_cut_time      = float(res['End_P_cut_time']) #After P arrival seconds
    sps                 = float(res['sps'])  #samples per seconds
    corr_window         = int(res['corr_window'])
    threshold_correlation=float(res['threshold_correlation'])
    SNR = float(res['SNR'])
    SNR_window         = int(res['SNR_window'])
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
    source_grid_extend_x  = float(res['source_grid_extend_x'])   #degrees
    source_grid_extend_y  = float(res['source_grid_extend_y'])   #degrees
    source_depth_size   = float(res['source_depth_size']) #km
    source_depth_extend = float(res['source_grid_extend']) #km
    try:
        opts, args = getopt.getopt(argv,"h:p:I:i:O:o:E:s:a:d:B:C:S:A:G:g:",["help=","processes=","Input_dir=","input_file=",\
                                                            "Output_dir=","output_file=","Exp_name=","sps=","azimuth_range="\
                                                                "distance_range=","Band_pass=","Correlation_thresh=",\
                                                                    "SNR=","Array_name=","Grid_extend=","grid_size="])
        #print(opts)
        #print(args)
    except getopt.GetoptError:
        print('\n###########################################################################################')
        print(' A simple run involves following where you specify your input_default.csv in the end.\n\
                Simple run example: python bp_streamed_parallel_prep_data_cmd_option.py input_default.csv\n\
                Below are the available options which you can pass as a command line argument:\n\
                -h : help\n\
                -p : no of parts to run in parallel (e.g., no of available cores). Default is 1\n\
                -I : input directory. Default is ./data/\n\
                -i : input file name in the data_tomo folder. Format x(*) y(*) depth(km) Vs(km/s)\n\
                -O : output directory. Default is ./output\n\
                -o : output file name which will be saved in output folder\n\
                -E : Experiment name. This will be used a\n\
                -s : Sampling rate i.e. SPS. Default is 20.\n\
                -a : Comma separated Azimuth range(-180/180) where first is the low and second is max (e.g., 60,90) \n\
                -d : Comma separated Distnace range in degrees where first is the low and second is max (e.g., 30,90)\n\
                -B : Comma separated min and max frequency (Hz) for the bandpass filter (0.2/5.0)\n\
                -C : Threshold cross-correlation coefficient(0-1.0) for waveform selection\n\
                -S : Signal to noise ratio.Default is 2\n\
                -G : Source grid extend in degrees. A square grid of this size centered at the hypocenter will be created.\n\
                -g : Source grid size in degrees.\n\
                -A : name of the array (e.g., AU,EU etc)\n\
                \nAll of the input options will be written in the input.csv file in the output directory.')
        print('###########################################################################################\n')   
        sys.exit(2)
    if (len(opts)!=0):
        for opt, arg in opts:
            #print(opt,arg)
            #if opt == '-h':
            if opt in ['-h','--help']:
                print('\n###########################################################################################')
                print(' A simple run involves following where you specify your input_default.csv in the end.\n\
                        Simple run example: python bp_streamed_parallel_prep_data_cmd_option.py input_default.csv\n\
                        Below are the available options which you can pass as a command line argument:\n\
                        -h : help\n\
                        -p : no of parts to run in parallel (e.g., no of available cores). Default is 1\n\
                        -I : input directory. Default is ./data/\n\
                        -i : input file name in the data_tomo folder. Format x(*) y(*) depth(km) Vs(km/s)\n\
                        -O : output directory. Default is ./output\n\
                        -o : output file name which will be saved in output folder\n\
                        -E : Experiment name. This will be used a\n\
                        -s : Sampling rate i.e. SPS. Default is 20.\n\
                        -a : Comma separated Azimuth range(-180/180) where first is the low and second is max (e.g., 60,90) \n\
                        -d : Comma separated Distnace range in degrees where first is the low and second is max (e.g., 30,90)\n\
                        -B : Comma separated min and max frequency (Hz) for the bandpass filter (0.2/5.0)\n\
                        -C : Threshold cross-correlation coefficient(0-1.0) for waveform selection\n\
                        -S : Signal to noise ratio.Default is 2\n\
                        -G : Source grid extend in degrees. A square grid of this size centered at the hypocenter will be created.\n\
                        -g : Source grid size in degrees.\n\
                        -A : name of the array (e.g., AU,EU etc)\n\
                        \nAll of the input options will be written in the input.csv file in the output directory.')
                print('###########################################################################################\n')   
                sys.exit()
            elif opt in ['-p', '--processes']:
                num_cores = int(arg)
                print('No of cores',num_cores)
            elif opt in ['-I', '--input_dir']:
                inputdir = str(arg)
            elif opt in ['-i', '--input_file']:
                input_file = arg
                print('Input file is:',input_file)
            elif opt in ['-O', '--output_dir']:
                outputdir =str(arg)
            elif opt in ['-o', '--output_file']:
                outputfile = arg
            elif opt in ['-E', '--Exp_name']:
                Event = str(arg)
                res['Event']=Event
                print('Event name is:',Event) 
            elif opt in ['-s', '--sps']:
                sps = float(arg)
                res['sps']=sps
                print('Sampling rate is:',sps)
            elif opt in ['-a', '--azimuth_range']:
                azimuth_range = arg.split(',')
                azimuth_min=float(azimuth_range[0])
                azimuth_max=float(azimuth_range[1])
                res['azimuth_min']=azimuth_min
                res['azimuth_max']=azimuth_max
                print('Azimuth range is:',azimuth_range)
                print('Minimum Azimuth is:',azimuth_min)
                print('Maximum Azimuth is:',azimuth_max)
            elif opt in ["-d", "--distance_range"]:
                distance_range = arg.split(',')
                dist_min=float(distance_range[0])
                dist_max=float(distance_range[1])
                res['dist_min']=dist_min
                res['dist_max']=dist_max
                print('Distance range is:',distance_range)
            elif opt in ['-B', '--Band_pass']:
                Band_pass = arg.split(',')
                bp_l=float(Band_pass[0])
                bp_u=float(Band_pass[1])
                res['bp_l']=bp_l
                res['bp_u']=bp_u
                print('Bandpass range is:',Band_pass)            
            elif opt in ['-C', '--Correlation_thresh']:
                threshold_correlation = float(arg)
                res['threshold_correlation']=threshold_correlation
                print('Correlation threshold is:',threshold_correlation)
            elif opt in ['-S','--SNR']:
                SNR = float(arg)
                res['SNR']=SNR
                print('Signal to noise ration is:',SNR) 
            elif opt in ["-A", "--Array_name"]:
                Array_name = str(arg)
                res['Array_name']=Array_name
                print('Array name is:',Array_name)                 
            elif opt in ["-G", "--Grid_extend"]:
                source_grid_extend = float(arg)
                res['source_grid_extend']=source_grid_extend
                print('Source grid extend is:',source_grid_extend)
            elif opt in ["-g", "--grid_size"]:
                source_grid_size = float(arg)
                res['source_grid_size']=source_grid_size
                print('Source grid size is:',source_grid_size)
            else:
                pass
        print('\n###########################################')
    else:
        print('\n###########################################')
        print('You did not provide required input.')
        print('Run the code with -h option for help.')
        print('###########################################\n')
    ##################################################################################
    # Main work here
    ##################################################################################
    path = os.getcwd()
    Exp_name=str(Array_name)+'_'+str(event_depth)+'km_'+str(res['model'])+'_'+str(res['threshold_correlation'])\
        +'_corr_'+str(source_grid_extend_x)+'_X'+str(source_grid_extend_y)+'_Y'+str(source_grid_size)+'_grid'+str(extra_label)
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
    ##########################################################################
    with open(outdir+'/'+'input.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in res.items():
            writer.writerow(row)
    ##########################################################################
    # Loading waveform data 
    # Note that the waveform data (miniseed) and stations (csv list) are downloaded from wilber 
    # (see https://ds.iris.edu/wilber3/find_event)
    # Also check out this http://eqseis.geosc.psu.edu/cammon/HTML/Classes/AdvSeismo/WLBR3/eventdatausingwilber3.html
    ##########################################################################
    stream_work= obspy.read(waveforms)
    stream_work=bp_lib.stream_info_populate(stream_work,stations,origin_time,event_depth,model)
    print("Total time taken: {:.1f} min".format((time.time()-time_start)/60.0))
    ##########################################################################
    # processing stream for distance,snr,azimuth and cutting
    ##########################################################################
    # SPS 
    print('Total no of traces before decimation criteria:', len(stream_work))
    stream_work = bp_lib.check_sps(stream_work,sps)
    print('Total no of traces after decimation criteria:', len(stream_work))
    
    ##########################################################################
    # CUtting before and after P arrival 
    ##########################################################################
    #stream_cut=stream_work.copy()
    print('Total no of traces before data gap checks:', len(stream_work))
    stream_work=bp_lib.stream_cut_P_arrival(stream_work,Start_P_cut_time,End_P_cut_time)
    print('Total no of traces after cutting and data gap checks ', len(stream_work))
    
    
    ######### distance
    print('Total no of traces before  distance criteria:', len(stream_work))
    stream_work = bp_lib.check_distance(stream_work,dist_min,dist_max)
    print('Total no of traces after distance criteria:', len(stream_work))

    ######### azimuth
    print('Total no of traces before  azimuth criteria:', len(stream_work))
    stream_work = bp_lib.check_azimuth(stream_work,azimuth_min,azimuth_max)
    print('Total no of traces after azimuth criteria:', len(stream_work))
  
    
    ######### removing mean and normalize
    print('Total no of traces before removing mean:', len(stream_work))
    stream_work= bp_lib.detrend_normalize_stream(stream_work,type='demean')
    print('Total no of traces after removing mean:', len(stream_work))
    
    ######### SNR check
    print('Total no of traces before  SNR criteria:', len(stream_work))
    stream_work = bp_lib.snr_check(stream_work,SNR,SNR_window,SNR_window)
    print('Total no of traces after SNR criteria:', len(stream_work))

    ##########################################################################
    # cross-correlation
    # Cross-correlation is perfpormed 2 times in order to keep the reference 
    # trace in the center of the array
    ##########################################################################
    Ref_station_index=bp_lib.get_ref_station(stream_work)
    ref_trace = stream_work[Ref_station_index]
    #temp=stream_work.copy()
    #for tr in stream_work:
    #    tr.filter('bandpass',freqmin=bp_l,freqmax=bp_u)
    #ref_trace=bp_lib.get_stream_stack(stream_work,model,event_depth,origin_time)
    #ref_trace.detrend(type='demean')
    #ref_trace.normalize()
    #print('Total no of traces before Cross-correlation:', len(stream_work))
    #print('Performning cross-correlation. Without filtering')
    #stream_work=bp_lib.crosscorr_stream_xcorr_no_filter(stream_work,ref_trace,Start_P_cut_time,corr_window,corr_window,corr_window,threshold_correlation)
    #stream_work=bp_lib.crosscorr_stream_xcorr(stream_work,ref_trace,Start_P_cut_time,corr_window,corr_window,corr_window,
    #                                          bp_l,bp_u,threshold_correlation)
    #stream_work=bp_lib.crosscorr_stream_xcorr_P_arrival(stream_work,ref_trace,corr_window,corr_window,corr_window,bp_l,bp_u,threshold_correlation)
    stream_work=bp_lib.crosscorr_stream_xcorr_no_filter_P_arrival(stream_work,ref_trace,corr_window,corr_window,corr_window,threshold_correlation)
    #stream_work=bp_lib.crosscorr_stream_prev(stream_work,ref_trace,corr_window,0)
    #stream_work=bp_lib.crosscorr_stream_xcorr(stream_work,ref_trace,corr_window,corr_window,corr_window,bp_l,bp_u,
    #                                                    threshold_correlation)
    print('Total no of traces after Cross-correlation:', len(stream_work))
    ##########################################################################
    # cross-correlation
    #Ref_station_index=bp_lib.get_ref_station(stream_work)
    #ref_trace = stream_work[Ref_station_index]
    #print('Total no of traces before Cross-correlation:', len(stream_work))
    #print('Performning cross-correlation. Without filtering')
    #stream_work=bp_lib.crosscorr_stream_xcorr_no_filter(stream_work,\
    #                                                    ref_trace,corr_window,corr_window,corr_window,threshold_correlation)
    #stream_work=bp_lib.crosscorr_stream_xcorr(stream_work,\
    #                                                    ref_trace,corr_window,corr_window,corr_window,
    #                                                    bp_l,bp_u,
    #                                                    threshold_correlation)
    print('Total no of traces after Cross-correlation:', len(stream_work))
  
    ##########################
    # final BB stream
    stream_for_bp=stream_work.copy()
    ##########################################################################
    # Making potential sources grid
    ##########################################################################
    #slong,slat          = bp_lib.make_source_grid(event_long,event_lat,source_grid_extend,source_grid_size)
    slong,slat          = bp_lib.make_source_grid_hetero(event_long,event_lat,source_grid_extend_x,source_grid_extend_y,source_grid_size)
    ##########################################################################
    print('Finished preparing data.')
    print("Total time taken: {:.1f} min".format((time.time()-time_start)/60.0))
    print('Now writing the stream and its info.')
    print('Writing the stream info in parallel.')
    results = Parallel(n_jobs=num_cores)(
        delayed(process_location)(j, slat, slong, stream_for_bp, event_depth, origin_time, model)
        for j in range(len(slat)) )
    beam_info=[];
    beam_info = np.concatenate(results)
    print("Total time taken: {:.1f} min".format((time.time()-time_start)/60.0))
    #####################################################
    # Writing the source info for the array
    np.save(outdir+'/'"beam_info",beam_info,allow_pickle=True)
    print('Writing the array info in parallel.')
    #####################################################
    # getting the array info e.g., station location, P_arrival, Correlation parametes etc.
    stream_info = bp_lib.save_stream_info(stream_for_bp)
    np.save(outdir+'/'"array_bp_info",stream_info,allow_pickle=True)
    #####################################################
    # Saving the array obspy stream
    print('Writing the stream.')
    stream_for_bp.write(outdir+'/'"stream.mseed")
    print("Total time taken: {:.1f} min".format((time.time()-time_start)/60.0))
    print('Now computing station weight..')
    stream_for_bp=bp_lib.stream_station_weight(stream_for_bp)
    print('Done computing station weight.')
    print("Data prepration DONE for Exp:",  outdir)
    print("Total time taken: {:.1f} min".format((time.time()-time_start)/60.0))
    print('#########################################################')
    print("Now to make the make by running following. Note that")
    print("the freuncy band from the input.csv file be used as default.")
    print("If you want to change it add the lower and higher frequency at the end separated by space.")
    print()
    print()
    print("Now run the following to use frequency and from input.csv: ")  
    print()
    print("python bp_streamed_parallel_make_beam.py ", outdir)  
    print()
    print()
    print("Run the following to use differnt frequency band: ")  
    print()
    print("python bp_streamed_parallel_make_beam.py", outdir ,"bp_l bp_u")  
    print()
    print("where bp_l is the lower end and bp_u is upper end of the frequency band. ")  

    '''
    print('Now making the beam...')
    ##########################################################################
    # Make beam
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
            cut = tr.data/np.max(np.abs(tr.data)) * tr.stats.Corr_coeff/tr.stats.Station_weight
            stack.append(cut[0:int((stack_start+stack_end)*sps)])
        beam.append(np.sum(stack,axis=0))
    ## saving
    print('Done making the beam.')
    print("Total time taken: {:.1f} min".format((time.time()-time_start)/60.0))
    print('Saving the beam.')
    file_save='beam_'+str(bp_l)+'_'+str(bp_u)+'_'+str(Array_name)+'.dat'
    np.savetxt(outdir+'/'+file_save,beam)
    print("Progress back-projection DONE for Exp:",  outdir)
    print("Total time taken: {:.1f} min".format((time.time()-time_start)/60.0))
    '''

if __name__ == '__main__':
    main(sys.argv[1:])