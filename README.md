# High-frequency Back Projection
This is a python implementation of https://github.com/ajay6763/earthquake_bp_obspy

The python code allows us to obtain a high-frequency back-projection image of high magnitude events (M_w>=7).

It is compatible with Python 3 and requires the following standard Python libraries:

1. obspy
2. numpy
3. pandas

## Data
Data is obtained from [IRIS Wilber](https://ds.iris.edu/wilber3/find_event) in Miniseed format. Data and station list must be stored in the ```./data``` directory. Note that the stations must be teleseismic with respect to the event (30-90 degrees).

## Data Preparation

Prepare data using [bp_streamed_parallel_prep_data_cmd_option](https://github.com/ajay6763/earthquake_bp_obspy/blob/main/bp_streamed_parallel_prep_data_cmd_option.py)

To get info about the input data format and options run the following in the ```earthquake_bp_obspy``` folder using the default input.csv file:

```
python3 bp_streamed_parallel_prep_data_cmd_option.py -h
```

This should give the following result:
```
python bp_streamed_parallel_prep_data_cmd_option.py -h

###########################################################################################
 Welcome for help run this script with -h option

You did not provided input file (.csv file). You must run without -h option to use the input_default.csv.
Do you want to continue with default input.csv? (yes/no) :yes

###########################################################################################
 A simple run involves following where you specify your input_default.csv in the end.
                Simple run example: python bp_streamed_parallel_prep_data_cmd_option.py input_default.csv
                Below are the available options which you can pass as a command line argument:
                -h : help
                -p : no of parts to run in parallel (e.g., no of available cores). Default is 1
                -I : input directory. Default is ./data/
                -i : input file name in the data_tomo folder. Format x(*) y(*) depth(km) Vs(km/s)
                -O : output directory. Default is ./output
                -o : output file name which will be saved in output folder
                -E : Experiment name. This will be used a
                -s : Sampling rate i.e. SPS. Default is 20.
                -a : Comma separated Azimuth range(-180/180) where first is the low and second is max (e.g., 60,90)
                -d : Comma separated Distnace range in degrees where first is the low and second is max (e.g., 30,90)
                -B : Comma separated min and max frequency (Hz) for the bandpass filter (0.2/5.0)
                -C : Threshold cross-correlation coefficient(0-1.0) for waveform selection
                -S : Signal to noise ratio.Default is 2
                -G : Source grid extend in degrees. A square grid of this size centered at the hypocenter will be created.
                -g : Source grid size in degrees.
                -A : name of the array (e.g., AU,EU etc)

All of the input options will be written in the input.csv file in the output directory.
###########################################################################################
```

Split the data into arrays (with respective latitute, longitude and azimuthal distance restrictions) and run the code for each array separately. Make necessary changes to the input_default.csv file accordingly.

A sample run would look like this:
```
python bp_streamed_parallel_prep_data_cmd_option.py -d 30,90 -a 20,50 -B 0.5,4 -G 2 -g 0.1 -A US -O US_output -i input_default.csv
```

Parameters which have not been specified will have default values. Each run would create a new directory (inside the earthquake_bp_obspy directory) for that particular array.

## Combining Arrays

After preparing data for adequate arrays, the data can be compined using [bp_combine_arrays](https://github.com/ajay6763/earthquake_bp_obspy/blob/main/bp_combine_arrays.py) and the names of the separate array directories as the parameters in the parameter list.

The array directory that leads the parameter list would be considered as the reference array. For example:

```
python bp_combine_arrays.py TWN_7.4_US_34.75km_iasp91_0.8_corr_0.1_grid TWN_7.4_EU_34.75km_iasp91_0.8_corr_0.1_grid TWN_7.4_AU_34.75km_iasp91_0.8_corr_0.1_grid
```
Here, the US array is considered as the reference array for this image.

The final image is generated with the first array as the reference array. It is stored in the ```./combined``` folder.
