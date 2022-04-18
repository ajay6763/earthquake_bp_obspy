import obspy
from obspy.clients.fdsn.mass_downloader import CircularDomain, RectangularDomain, Restrictions, MassDownloader
from obspy.taup import TauPyModel
import pandas as pd


######################################
# Event info
origin_time = obspy.UTCDateTime(2016, 1, 3, 23, 5, 22)
event_lat=24.80360
event_long=93.65050
event_depth=55.0 # km

#######################
## This in case we make a list of stations array wise or in total such then we can ask specifically for data
#data = pd.read_csv('./EU_wilber-stations.txt', sep='|')
#data['Net'].unique()
Array='AU_test'


#####################################
# A typical data download in obspy results in a stations info in xml files that have all the station info including 
# response files etc. The actual waveform data is downloaded separately. This mean when we want to work with waveform 
# data we have to read the station files.
# Drive location to save wavefrom data
mseed_storage = "./data/test/bp_waveforms_"+str(Array)
# Drive location to save station inventory
stationxml_storage = "./data/test/bp_stations_"+str(Array)



###########################
# This stuff defines restrictions on the data for time
start_time_waveform_data=origin_time
end_time_waveform_data=origin_time + 60*60# 3600 seconds


#model = TauPyModel(model="ak135")
#min_travel_time = model.get_travel_times(source_depth_in_km=event_depth,
#                                          distance_in_degree=20,
#                                          phase_list=["P"])
#print(min_travel_time)
#arr = min_travel_time[0]
#starttime = origin_time #+ arr.time - 60.0  # 1 minute before the event

#max_travel_time = model.get_travel_times(source_depth_in_km=event_depth,
#                                          distance_in_degree=95,
#                                          phase_list=["P"])
#print(max_travel_time)

#endtime = origin_time + 800 + (10.0*60.0)  # 10 minutes after the event
#endtime = origin_time + 800 + (10.0*60.0)  # 10 minutes after the event


##########################################################
# This is location from where you want to get the data
# IN principal 
# Circular domain around the epicenter. This will download all data between
# 70 and 90 degrees distance from the epicenter. This module also offers
# rectangular and global domains. More complex domains can be defined by
# inheriting from the Domain class.
#domain = CircularDomain(latitude=event_lat, longitude=event_long,minradius=30.0, maxradius=90.0)
#### For AU
domain = RectangularDomain(minlatitude=-50, maxlatitude=5,
                           minlongitude=85, maxlongitude=173)
#### For EU
#domain = RectangularDomain(minlatitude=25, maxlatitude=65,
#                           minlongitude=-20, maxlongitude=45)

#### For AF
#domain = RectangularDomain(minlatitude=25, maxlatitude=-40,
#                           minlongitude=-19, maxlongitude=64)

#### For JP
#domain = RectangularDomain(minlatitude=23, maxlatitude=62,
#                           minlongitude=120, maxlongitude=168)

restrictions = Restrictions(
    # Get data from origin time of the event to one hour after the
    # event. This defines the temporal bounds of the waveform data.
    starttime=origin_time,
    endtime=origin_time + 60*60,
    # You might not want to deal with gaps in the data. If this setting is
    # True, any trace with a gap/overlap will be discarded.
    reject_channels_with_gaps=True,
    # And you might only want waveforms that have data for at least 95 % of
    # the requested time span. Any trace that is shorter than 95 % of the
    # desired total duration will be discarded.
    minimum_length=1.00,
    ## Network
    # For Australia 
    #network="2P,3M,4N,6C,7G,8J,AU,DU,S1,YS",
    # For Africa
    #network="AF,G,IU,GE,G,II,KV,ZT,NR,GT,8A",
    # For Japan
    #network="CB,IU,KS,KG,JP,G,II",
    # FOr EU
    #network="KZ,II,MD,IU,UD,IM,RO,GE,HE,EE,FN,PL,NO,1G,SJ,UP,MN,HU,CZ,OE,Z3,CR,HF,SL,GR,DK,SX,TH,BW,NI,OX,IV,SI,CH,GU,NR,RN,NL,FR,G,RD,BE,GB,UK,BN,CA,EI,VI,WM,2M",
    # No two stations should be closer than 10 km to each other. This is
    # useful to for example filter out stations that are part of different
    # networks but at the same physical station. Settings this option to
    # zero or None will disable that filtering.
    minimum_interstation_distance_in_m=0.0, #10E3,
    # Only HH or BH channels. If a station has HH channels, those will be
    # downloaded, otherwise the BH. Nothing will be downloaded if it has
    # neither. You can add more/less patterns if you like.
    #channel_priorities=["HH[Z]", "BH[Z]", "LH[Z]", "SH[Z]"],
    channel_priorities=["HH[Z]", "BH[Z]", "LH[Z]", "SH[Z]"],

    # Location codes are arbitrary and there is no rule as to which
    # location is best. Same logic as for the previous setting.
    location_priorities=["", "00", "10"])


# No specified providers will result in all known ones being queried.
mdl = MassDownloader()


mdl.download(domain, restrictions,threads_per_client=3, mseed_storage=mseed_storage,
             stationxml_storage=stationxml_storage)
