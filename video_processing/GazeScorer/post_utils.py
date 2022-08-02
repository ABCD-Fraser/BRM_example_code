import numpy as np
import pandas as pd
from statsmodels.stats import inter_rater


from GazeScorer import settings_utils


def proportions(etdset, eye='overall', specify_measure=None):

    """Calculates proportion of time spent looking at each location

        Calculates proportion of time spent looking at each location. The default is to use the overall data, but can be run on a single eye. 
        It is also possible to specify whether Time, Proportion, or Both are provided

    Args:
        etdset (dataframe): data frame of eye tracking data output from GazeScorer
        eye (str): string defining which eye to use. Default = overall
        specify_measure (str): string defining which measure to output ('time' or 'proportion'). None=both

    Return:
        proportion_output: single row dataframe with desired outputs
    """
    # Selects eye index for desired eye
    if eye == 'overall':
        eye_index = 'agazeraw'
    elif eye == 'left':
        eye_index = 'algazeraw'
    elif eye == 'right':
        eye_index = 'argazeraw'
    else:
        print('Incorrect eye eye_index provided. Will proceed with overall estimate')
        eye_index = 'agazeraw'

    # Gets the number of frames and duration of video
    num_frames = len(etdset['frame_timestamp'])
    stop_time = etdset['frame_timestamp'].iloc[-1]
    
    # Gets number of Left frames and caluclates time and proportion
    num_l = len(etdset[etdset[eye_index]==1])
    prop_l = num_l/num_frames
    time_l = (stop_time * prop_l) * 1000

    # Gets number of Right frames and caluclates time and proportion
    num_r = len(etdset[etdset[eye_index]==-1])
    prop_r = num_r/num_frames
    time_r = (stop_time * prop_r) * 1000

    # Gets number of Centre frames and caluclates time and proportion
    num_c = len(etdset[etdset[eye_index]==0])
    prop_c = num_c/num_frames
    time_c = (stop_time * prop_c) * 1000

    # Updates the output based on specify_measures input
    if specify_measure == 'proportion':
        proportion_output = pd.DataFrame({'proportion_centre': prop_c,'proportion_left': prop_l, 'proportion_right': prop_r}, index=[0])
    elif specify_measure == 'time':
        proportion_output = pd.DataFrame({'time_centre': time_c,'time_left': time_l, 'time_right': time_r}, index=[0])
    else:
        proportion_output = pd.DataFrame({'proportion_centre': prop_c,'proportion_left': prop_l, 'proportion_right': prop_r, 
    'time_centre': time_c,'time_left': time_l, 'time_right': time_r}, index=[0])

    return proportion_output


def firstlook(etdset, eye='overall', fix_buffer=5, n_shifts=False):

    """Ouputs the first look direction

        Identifies the first look direction and provides the timestamp and duration (plus optionally number of shifts overall).
        The first look is defined by the first direction shift that is consistant accross a number of frames defined by the fix_buffer variable.

    Args:
        etdset (dataframe): data frame of eye tracking data output from GazeScorer
        eye (str): string defining which eye to use. Default = overall
        fix_buffer (int): Defines the min number of frames required to define a look
        n_shifts (bool): boolean to identify if number of shifts should be reported.

    Return:
        firstlook_output: single row dataframe with desired outputs
    """

    # Runs gaze_shift data to get number of shifts and timestamps
    gazeshift_data = gaze_shift(etdset, eye, fix_buffer) 

    # Get direction, timestamo and duration from the gaze_shift data
    # if only 1 look calculate duration from length of video
    if len(gazeshift_data) == 2:
        firstlook_output = pd.DataFrame({'direction': gazeshift_data['direction'].iloc[1], 
                                            'timestamp': gazeshift_data['timestamp'].iloc[1], 
                                                'duration': etdset['frame_timestamp'].iloc[-1] - gazeshift_data['timestamp'].iloc[1]}, 
                                                 index=[0])
    # Calculate duration from previous 
    elif len(gazeshift_data) > 1:
        firstlook_output = pd.DataFrame({'direction': gazeshift_data['direction'].iloc[1], 
                                            'timestamp': gazeshift_data['timestamp'].iloc[1], 
                                                'duration': gazeshift_data['timestamp'].iloc[2] - gazeshift_data['timestamp'].iloc[1]}, 
                                                    index=[0])
    # No looks fills with placeholders/NA  
    else:
        firstlook_output = pd.DataFrame({'direction': 'None', 'timestamp': float(0.0), 'duration': pd.NA}, index=[0])

    # if n_shifts are requested add maximum number of gaze shifts.     
    if n_shifts:
        firstlook_output['n_shifts'] = gazeshift_data['n_shifts'].iloc[-1]

    return firstlook_output

def gaze_shift(etdset, eye='overall', fix_buffer=5):

    """Ouputs a dataframe of each gaze shift accross the course of the video analysis

        Identifies each change in gaze orientation and provides the timestamp, duration, and current number of gaze shifts.
        A look is defined by the a direction shift that is consistant accross a number of frames defined by the fix_buffer variable.

    Args:
        etdset (dataframe): data frame of eye tracking data output from GazeScorer
        eye (str): string defining which eye to use. Default = overall
        fix_buffer (int): Defines the min number of frames required to define a look
        
    Return:
        gazeshift_output: dataframe with desired outputs
    """
    # Selects eye index for desired eye
    if eye == 'overall':
        eye_index = 'agazeraw'
    elif eye == 'left':
        eye_index = 'algazeraw'
    elif eye == 'right':
        eye_index = 'argazeraw'
    else:
        print('Incorrect eye index provided. Will proceed with overall estimate')
        eye_index = 'agazeraw'

    # creates empy variables for outputs and for iterating through variables
    output_data = {'direction': [], 'timestamp': []}
    previous_val = 0
    count = 0
    skip = False
    
    # iterate through data set 
    for i, row in etdset.iterrows():
        
        # sets current gaze direction
        current_value = row[eye_index]
       
        # Selects if the current value == previous value and 
        if previous_val == current_value:
            # If values match then select direction value and add to the count for that value. 
            if current_value == 0:
                previous_val = current_value
                count += 1
                # If number of rows with this direction > fix_buffer add direction and timestamp to the outoput
                if count == fix_buffer:
                    if skip:
                        skip = False
                    else:
                        output_data['direction'].append('centre')
                        output_data['timestamp'].append(row['frame_timestamp']*1000)
                # if its the first iteration set the value and ignore the next frame. 
                elif i == 0:
                    output_data['direction'].append('centre')
                    output_data['timestamp'].append(row['frame_timestamp']*1000)
                    skip = True
            
            # If values match then select direction value and add to the count for that value.
            elif current_value == 1:
                previous_val = current_value
                count += 1
                # If number of rows with this direction > fix_buffer add direction and timestamp to the outoput
                if count == fix_buffer:
                    if skip:
                        skip = False
                    else:
                        output_data['direction'].append('right')
                        output_data['timestamp'].append(row['frame_timestamp']*1000)
                # if its the first iteration set the value and ignore the next frame.
                elif i == 0:
                    output_data['direction'].append('left')
                    output_data['timestamp'].append(row['frame_timestamp']*1000)
                    skip = True

            # If values match then select direction value and add to the count for that value.
            elif current_value == -1:
                previous_val = current_value
                count += 1
                # If number of rows with this direction > fix_buffer add direction and timestamp to the outoput
                if count == fix_buffer:
                    if skip:
                        skip = False
                    else:
                        output_data['direction'].append('left')
                        output_data['timestamp'].append(row['frame_timestamp']*1000)
                # if its the first iteration set the value and ignore the next frame.
                elif i == 0:
                    output_data['direction'].append('right')
                    output_data['timestamp'].append(row['frame_timestamp']*1000)
                    skip = True
        # if previosu value doesnt match set previosu value and reset count
        else:
            previous_val = current_value
            count = 0 
    
    # Set column with range to match mumber of gaze shifts to output
    output_data['n_shifts'] = range(len(output_data['direction']))
    gazeshift_output = pd.DataFrame(output_data)

    return gazeshift_output    
        


def update_cols2proc(etdset, cols2proc):

    """Updates which columns need to be processed when preparing the etdset
    """

    # checks if timstamps has been added and appends if so
    if "h_location" in etdset:
        cols2proc = cols2proc + ("h_location", "h_movement")
    if "v_location" in etdset:
        cols2proc = cols2proc + ("v_location", "v_movement")

    # checks if manual scoring has been done and appends if so
    if "sgazeraw" in etdset:
        cols2proc = cols2proc + ("sgazeraw",)
    if "slgazeraw" in etdset:
        cols2proc = cols2proc + ("slgazeraw",)
    if "srgazeraw" in etdset:
        cols2proc = cols2proc + ("srgazeraw",)

    return cols2proc

def rename_cols(etdset):

    """Too update - Takes in etdset and renames columns named in the processing stage. Returns updated etdset 
    """

    colsrenamed = {
        "elhdirval": "algazeraw",
        "erhdirval": "argazeraw",
        "hdirval": "agazeraw",
        "ildx": "algazeraw_dx",
        "irdx": "argazeraw_dx",
        "hdirdx": "agazeraw_dx",
    }

    etdset.rename(columns=colsrenamed, inplace=True)

    return etdset

def prepare_output(etdset, output_level="basic", PID=None):

    """Updates the etdset columns to the desired level

        Selects which columns are needed based on the output_level. The basic output will give the frame timestamp and the gaze
        orientation score and normalised values. Intermidiate will add measurments for the eyes and LKV data. Full provides 
        all data, including the xy coords of the face. 

    Args
        output_level (str): Sets the level of output of the etdset (basic (default), intermediate, or full)
        PID (str): Participant identifier to be added if requested
    
    Returns
        etdset (dataframe): returns prepared etdset
    """

    # Renames columnas (TO BE UPDATED)
    etdset = rename_cols(etdset)

    if output_level == "full":
        # Sets columns that will be processed at this level 
        cols2proc = etdset.columns       
        
    elif output_level == "intermediate":
        # Sets columns that will be processed at this level
        cols2proc = (
            "frame_timestamp",
            "fw",
            "fh",
            "elw",
            "elh",
            "erw",
            "erh",
            "ilx",
            "ily",
            "ilr",
            "irx",
            "iry",
            "irr",
            "elaratio",
            "elxmidraw",
            "elymidraw",
            "eraratio",
            "erxmidraw",
            "erymidraw",
            "aratioavg",
            "elxmid",
            "algazeraw_dx",
            "elxmidbuf",
            "elxmidmin",
            "elxmidmax",
            "erxmid",
            "argazeraw_dx",
            "erxmidbuf",
            "erxmidmin",
            "erxmidmax",
            "algazeraw",
            "argazeraw",
            "agazeraw",
            "hdirx",
            "agazeraw_dx",
            "alconf",
            "alnlkv",
            "allkvlen",
            "arconf",
            "arnlkv",
            "arlkvlen",
            "aconf",
            "anlkv",
            "alkvlen",

        )

    else:
        # Default - Sets columns that will be processed at this level
        cols2proc = (
            "frame_timestamp",
            "algazeraw",
            "argazeraw",
            "agazeraw",
            "algazeraw_dx",
            "argazeraw_dx",
            "agazeraw_dx",
        )


    #updates columns to process if additional data has been added
    cols2proc = update_cols2proc(etdset, cols2proc)

    # Renames columnas (TO BE UPDATED)
    etdset = rename_cols(etdset)

    # Selects columns 
    etdset = etdset.loc[:, cols2proc]

    # adds Participant ID if provided
    if PID is not None:
        etdset["participant"] = PID

    # Appends a range for list of frames
    etdset['frames'] = range(1, (len(etdset)+1))

    return etdset



def lkv_process(etdset, settings=None, output_level='basic'):

    """Applys a LKV process to the eye tracking data to remove blinks and other missing data from the GazeScorer data. Adjusts confidence varibles as needed
    """

    # imports settings if not provided
    if settings is None:
        settings = settings_utils.assign_settings()

    #Establishes required settings variables    
    proc = settings["proc"]
    gaze_set = settings["gaze"]
    plots = settings["plots"]

    # Sets up markers for each gaze location and for each eye for processing.  
    gazescores = {"middle": gaze_set["hdirs"]["MIDDLE"], "left": gaze_set["hdirs"]["LEFT"],
                    "right": gaze_set["hdirs"]["RIGHT"], "blink": gaze_set["hdirs"]["BLINK"], 
                    "unknown": gaze_set["hdirs"]["UNKNOWN"]}
    eyes = gaze_set['elist_short']

    # Variables for establishing confidence and which eye is dominant
    confscores = {"noconfidence": 0, "fullconfidence": 1}
    sources = {"either": 0, "left eye": 1, "right eye": 2}
    
    # create 
    dset = etdset.copy()

    prow = []

    # process each frame to replace blinks with last good value in manual and automatic scoring
    # each frame has a gaze either left/middle/right, automatic scoring decreases confidence
    blinks = {
        "al": False,
        "ar": False,
        "a": False,
        "sl": False,
        "sr": False,
        "s": False,
    }
    for irow, row in dset.iterrows():
        for eye in eyes:
            # automated scoring
            gaze, conf, nlkv, lkvlen, gazex, _ = init_vals(
                gazescores, confscores)

            if row["a" + eye + "gazeraw"] == gazescores["blink"]:
                blinks["a" + eye] = True
                if irow == 0:
                    # no last known value the only reasonable estimate is middle
                    gaze = gazescores["middle"]
                    conf = confscores["noconfidence"]
                    nlkv = 1
                    gazex = 0
                else:
                    gaze = prow["a" + eye + "gazeraw"]
                    conf = confscores["noconfidence"]
                    nlkv = prow["a" + eye + "nlkv"] + 1
                    gazex = prow["a" + eye + "gazex"]
            else:
                if blinks["a" + eye] and irow > 0:
                    lkvlen = dset.loc[irow - 1, "a" + eye + "nlkv"]
                blinks["a" + eye] = False
                gaze = row["a" + eye + "gazeraw"]
                conf = confscores["fullconfidence"]
                nlkv = 0
                if gaze == gazescores["middle"]:
                    gazex = 0
                else:
                    gazex = row["a" + eye + "gazeraw_dx"]

            dset.loc[irow, "a" + eye + "gazeraw"] = gaze
            dset.loc[irow, "a" + eye + "conf"] = conf
            dset.loc[irow, "a" + eye + "nlkv"] = nlkv
            dset.loc[irow, "a" + eye + "lkvlen"] = lkvlen
            dset.loc[irow, "a" + eye + "gazex"] = gazex

        # overall gaze: automatic scoring
        gaze, conf, nlkv, lkvlen, gazex, source = init_vals(
            gazescores, confscores)
        algaze = dset.loc[irow, "algazeraw"]
        argaze = dset.loc[irow, "argazeraw"]
        if algaze == argaze:
            # eyes agree
            gaze = algaze
            conf = dset.loc[irow, "alconf"] + dset.loc[irow, "arconf"]
            nlkv = np.min([dset.loc[irow, "alnlkv"],
                            dset.loc[irow, "arnlkv"]])
            # for gazex always use temporal excursion
            if gaze == gazescores["middle"]:
                gazex = 0
                source = sources["either"]
            elif gaze == gazescores["left"]:
                gazex = dset.loc[irow, "algazex"]
                source = sources["left eye"]
            else:
                gazex = dset.loc[irow, "argazex"]
                source = sources["right eye"]
        else:
            # eyes disagree
            if ((algaze == gazescores["right"]) and (argaze == gazescores["left"])) or (
                (algaze == gazescores["right"]) and argaze == gazescores["left"]):

                # eyes crossing or diverging
                gaze = gazescores["middle"]
                conf = confscores["noconfidence"]
                nlkv = np.min([dset.loc[irow, "alnlkv"],
                                dset.loc[irow, "arnlkv"]])
                gazex = 0
                source = sources["either"]
            elif (algaze == gazescores["middle"]) and (argaze == gazescores["right"]):
                # left eye is middle and right eye is right, use right eye (temporal gaze)
                gaze = argaze
                conf = dset.loc[irow, "arconf"]
                nlkv = dset.loc[irow, "arnlkv"]
                gazex = dset.loc[irow, "argazex"]
                source = sources["right eye"]
            elif (argaze == gazescores["middle"]) and (algaze == gazescores["left"]):
                # right eye is middle and left eye is left, use left eye (temporal gaze)
                gaze = algaze
                conf = dset.loc[irow, "alconf"]
                nlkv = dset.loc[irow, "alnlkv"]
                gazex = dset.loc[irow, "algazex"]
                source = sources["left eye"]
            else:
                # default middle
                gaze = gazescores["middle"]
                conf = confscores["noconfidence"]
                nlkv = np.max([dset.loc[irow, "alnlkv"],
                                dset.loc[irow, "arnlkv"]])
                gazex = 0
                source = sources["either"]
        if conf < 2:
            blinks["a"] = True
        else:
            if blinks["a"] and irow > 0:
                lkvlen = dset.loc[irow - 1, "anlkv"]
            blinks["a"] = False

        dset.loc[irow, "agazeraw"] = gaze
        dset.loc[irow, "aconf"] = conf
        dset.loc[irow, "anlkv"] = nlkv
        dset.loc[irow, "alkvlen"] = lkvlen
        dset.loc[irow, "agazex"] = gazex
        dset.loc[irow, "asource"] = source

        prow = dset.iloc[irow, :].copy()


    if output_level != 'full':
        
        etdset['algazeraw'] = dset['algazeraw']
        etdset['argazeraw'] = dset['argazeraw']
        etdset['agazeraw'] = dset['agazeraw']

        return etdset
    else:

        return dset


def init_vals(gazescores, confscores):

    """Initiates values for the LKV processing
    """

    gaze = gazescores["unknown"]
    conf = confscores["noconfidence"]
    nlkv = 0
    lkvlen = np.nan
    gazex = 0
    source = np.nan

    return gaze, conf, nlkv, lkvlen, gazex, source


def add_scoring(etdset, settings=None, left_eye=None, right_eye=None, overall_scoring=None, output_level='intermediate', apply_lkv=True):

    """Adds manually scored data to the eye tracking data set for comparison. Will also apply a LKV process by default. 
    """

    if left_eye is not None:
        if len(left_eye) != len(etdset):
            print(
                "Length of left eye data does not match the length of the data set and will be skipped"
            )
            left_eye = None
        else:
            etdset["slgazeraw"] = left_eye
            left=True
            # left_eye.copy()

    if right_eye is not None:
        if len(right_eye) != len(etdset):
            print(
                "Length of right eye data does not match the length of the data set and will be skipped"
            )
            right_eye = None
        else:
            # temp_etdset["srgazeraw"] = right_eye.copy()
            etdset["srgazeraw"] = right_eye
            right=True

    if overall_scoring is not None:
        if len(overall_scoring) != len(etdset):
            print(
                "Length of overall_scoring data does not match the length of the data set and will be skipped"
            )
            overall_scoring = None
        else:
            # temp_etdset["sgazeraw"] = overall_scoring.copy()
            etdset["sgazeraw"] = overall_scoring
            overall=True

    if apply_lkv:
        etdset = lkv_process_scoring(etdset, left=left, right=right, overall=overall, output_level=output_level, settings=settings)

    return etdset


def lkv_process_scoring(dsetraw, left=False, right=False, overall=False, output_level='basic', settings=None):

    """Applys a LKV process to the eye tracking data to remove blinks and other missing data from the GazeScorer data. Adjusts confidence varibles as needed
    """

     # imports settings if not provided
    if settings is None:
        settings = settings_utils.assign_settings()

    #Establishes required settings variables    
    proc = settings["proc"]
    gaze_set = settings["gaze"]
    plots = settings["plots"]

    # Sets up markers for each gaze location and for each eye for processing.  
    gazescores = {"middle": gaze_set["hdirs"]["MIDDLE"], "left": gaze_set["hdirs"]["LEFT"],
                    "right": gaze_set["hdirs"]["RIGHT"], "blink": gaze_set["hdirs"]["BLINK"], 
                    "unknown": gaze_set["hdirs"]["UNKNOWN"]}
    
    eyes = []
    if left:
        eyes.append('l')
    if right:
        eyes.append('r')
    if overall:
        eyes.append('')

    # Variables for establishing confidence and which eye is dominant
    confscores = {"noconfidence": 0, "fullconfidence": 1}
    sources = {"either": 0, "left eye": 1, "right eye": 2}

    acols = [
        "algaze",
        "alconf",
        "alnlkv",
        "allkvlen",
        "algazex",
        "argaze",
        "arconf",
        "arnlkv",
        "arlkvlen",
        "argazex",
        "agaze",
        "aconf",
        "anlkv",
        "alkvlen",
        "agazex",
        "asource",
    ]

    # procdset = pd.DataFrame()

    dset = dsetraw.copy()

    prow = []

    # process each frame to replace blinks with last good value in manual and automatic scoring
    # each frame has a gaze either left/middle/right, automatic scoring decreases confidence
    blinks = {
        "al": False,
        "ar": False,
        "a": False,
        "sl": False,
        "sr": False,
        "s": False,
    }
    for irow, row in dset.iterrows():
        for eye in eyes:
            # automated scoring
            gaze, conf, nlkv, lkvlen, _, _ = init_vals(
                gazescores, confscores)

            if row["s" + eye + "gazeraw"] == gazescores["blink"]:
                blinks["s" + eye] = True
                if irow == 0:
                    # no last known value the only reasonable estimate is middle
                    gaze = gazescores["middle"]
                    conf = confscores["noconfidence"]
                    nlkv = 1
                    
                else:
                    gaze = prow["s" + eye + "gazeraw"]
                    conf = confscores["noconfidence"]
                    nlkv = prow["s" + eye + "nlkv"] + 1

            else:
                if blinks["s" + eye] and irow > 0:
                    lkvlen = dset.loc[irow - 1, "s" + eye + "nlkv"]
                blinks["s" + eye] = False
                gaze = row["s" + eye + "gazeraw"]
                conf = confscores["fullconfidence"]
                nlkv = 0
                

            dset.loc[irow, "s" + eye + "gazeraw"] = gaze
            dset.loc[irow, "s" + eye + "conf"] = conf
            dset.loc[irow, "s" + eye + "nlkv"] = nlkv
            dset.loc[irow, "s" + eye + "lkvlen"] = lkvlen
        

        prow = dset.iloc[irow, :].copy()


    if output_level != 'full':
        
        dsetraw['slgazeraw'] = dset['slgazeraw']
        dsetraw['srgazeraw'] = dset['srgazeraw']
        dsetraw['sgazeraw'] = dset['sgazeraw']

        return dsetraw
    else:

        return dset

def add_timestamp(etdset,
                      h_event_start=None,
                      h_location_key=None,
                      h_movement_key=None,
                      v_event_start=None,
                      v_location_key=None,
                      v_movement_key=None,
                      ):
    
    """Calculates the targets location based on the timestamps and appends a key for each frame to the data frame.

    Args:
        etdset (dataframe): The current data processed data set
        h_event_start (list): optional - timepoint corrosponding to a new target location event for horizontal movements
        h_location_key (list): optional - location key that corresponds to the target horizontal location. Length must match h_event_start.
        h_movement_key (list): optional - key to mark if target is moving horizontally during the target event. Length must match h_event_start.
        v_event_start (list): optional - timepoint corrosponding to a new target location event for vertically movements
        v_location_key (list): optional - location key that corresponds to the target vertically location. Length must match v_event_start.
        v_movement_key (list): optional - key to mark if target is moving vertically during the target event. Length must match v_event_start.

    Returns:
        etdset (dataframe): The current data processed data set with appended timestamps
    """
    # Checks that length of horizontal events and keys provided match
    if h_event_start is not None:
        if h_location_key is None:
            print("Need to provide horizontal location key")
        elif len(h_event_start) != len(h_location_key):
            print(
                f"List of movment keys (length={len(h_location_key)}) is not the same length as list of horizontal events (length={len(h_event_start)})"
            )

        if h_movement_key is not None:
            if len(h_movement_key) != len(h_event_start):
                print(
                    f"List of movment keys (length={len(h_movement_key)}) is not the same length as list of horizontal events (length={len(h_event_start)})"
                )

    # Checks that length of vertical events and keys provided match
    if v_event_start is not None:
        if v_location_key is None:
            print("Need to provide horizontal location key")
        elif len(v_event_start) != len(v_location_key):
            print(
                f"List of movment keys (length={len(v_location_key)}) is not the same length as list of vertical events (length={len(v_event_start)})"
            )

        if v_movement_key is not None:
            if len(v_movement_key) != len(v_event_start):
                print(
                    f"List of movment keys (length={len(v_movement_key)}) is not the same length as list of vertical events (length={len(v_event_start)})"
                )

    if h_event_start is not None:

        # empty variables for horizontal and vertical score
        location_key = []
        movement_key = []

        # Creates empty data frame and adds event start points
        timepoint_df = pd.DataFrame()
        timepoint_df["event_start"] = list(h_event_start)

        # Creates even end list by taking next start event in list. append inf value as final value. 
        event_end = list(h_event_start)
        event_end = event_end[1:]
        event_end.append(float("inf"))
        timepoint_df["event_end"] = event_end

        # Adds horizontal location keys to the dataframe. Also add movement key if provided
        timepoint_df["key"] = h_location_key
        if h_movement_key is not None:
            timepoint_df["moving"] = h_movement_key

        
        # starts loop to write stimuli location
        for frame_row in etdset["frame_timestamp"]:
            
            # creates an index of the event that is occuring on the row
            event_index = timepoint_df[
                (timepoint_df["event_start"] <= float(frame_row))
                & (timepoint_df["event_end"] > float(frame_row))
            ]

            
            #appends locataion key to the row and movement key if provided
            location_key.append(event_index["key"].values[0])
            if h_movement_key is not None:
                movement_key.append(event_index["moving"].values[0])

        # Once completed location keys are added to the main etdset
        etdset["h_location"] = location_key

        #if provided movement key added to main etdset
        if h_movement_key is not None:
            etdset["h_movement"] = movement_key

    if v_event_start is not None:

        # empty variables for horizontal and vertical score
        location_key = []
        movement_key = []

        # Creates empty data frame and adds event start points
        timepoint_df = pd.DataFrame()
        timepoint_df["event_start"] = list(v_event_start)

        # Creates even end list by taking next start event in list. append inf value as final value. 
        event_end = list(v_event_start)
        event_end = event_end[1:]
        event_end.append(float("inf"))
        timepoint_df["event_end"] = event_end

        # Adds horizontal location keys to the dataframe. Also add movement key if provided
        timepoint_df["key"] = v_location_key
        if h_movement_key is not None:
            timepoint_df["moving"] = v_movement_key

        # starts loop to write stimuli location
        for frame_row in etdset["frame_timestamp"]:

            # creates an index of the event that is occuring on the row
            event_index = timepoint_df[
                (timepoint_df["event_start"] <= float(frame_row))
                & (timepoint_df["event_end"] > float(frame_row))
            ]

            #appends locataion key to the row and movement key if provided
            location_key.append(event_index["key"].values[0])
            if v_movement_key is not None:
                movement_key.append(event_index["moving"].values[0])

        # Once completed location keys are added to the main etdset
        etdset["v_location"] = location_key

        #if provided movement key added to main etdset
        if v_movement_key is not None:
            etdset["v_movement"] = movement_key

    return etdset

def ckap(etdset, eye=None, matrix=False):

    """Function to retrieve Cohens kappa score and upper/lower 95% CI for comparisons between gazescorer and an alternative"""


    # Selects eye index for desired eye
    if eye == 'overall':
        a_index = ['agazeraw']
        s_index = ['sgazeraw']
        o_index = ['overall']
    elif eye == 'left':
        a_index = ['algazeraw']
        s_index = ['slgazeraw']
        o_index = ['left']
    elif eye == 'right':
        a_index = ['argazeraw']
        s_index = ['srgazeraw']
        o_index = ['right']
    else:
        a_index = ['agazeraw', 'algazeraw', 'argazeraw']
        s_index = ['sgazeraw', 'slgazeraw', 'srgazeraw']
        o_index = ['overall', 'left', 'right']

    ckap_output = pd.DataFrame()

    ma_index = ['a_right', 'a_centre', 'a_left']
    ms_index = ['s_right', 's_centre', 's_left']

    for i in range(len(a_index)):

        # gets the two columns to compare
        agaze = etdset[a_index[i]].replace(-1,2)
        sgaze = etdset[s_index[i]].replace(-1,2)

        # agaze = agaze.replace(0,3)
        # sgaze = sgaze.replace(0,3)

        
        # concnats to new df
        irr_array = pd.concat([sgaze, agaze], axis=1)


        # Converts to a table for CK analysis
        irr_array = inter_rater.to_table(irr_array, bins=3)
        

        table = pd.DataFrame(irr_array[0])
        table.columns = ma_index
        table.index = ms_index

        # runs ck 
        ckap = inter_rater.cohens_kappa(irr_array[0], weights=None, wt=None)

        temp_ckap = pd.DataFrame({f'k_{o_index[i]}': ckap.kappa, 
        f'CI_low_{o_index[i]}': ckap.kappa_low, 
        f'CI_upp_{o_index[i]}': ckap.kappa_upp}, index=[0])

        # adds required variables to a DataFrame 
        ckap_output = pd.concat([ckap_output, temp_ckap], axis=1)

    if matrix:
        return ckap_output, table
    else:
        return ckap_output