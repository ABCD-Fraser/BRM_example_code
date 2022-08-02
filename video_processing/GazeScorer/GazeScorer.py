from GazeScorer import plot_utils
from GazeScorer import settings_utils
from GazeScorer import post_utils
from GazeScorer import video_utils


class process_video:

    """Process raw videos and provides methods for post-processing.

    This class is used for processing raw video footage to analyse eye gaze movements and estimating gaze orientation.
    To do this it begins by preprocessing and preparing the video for analyis, it then breaks the video to individual frames 
    before processing each frame individually. This produces an Object which holds the the eye tracking data set (etdset).
    From here, additional post-processing functions are available to summarise the data into more usable outputs. 
    These include: Proportion of looking time (Time and proportion of time attending to each location), 
    Firstlook (Direction, Timestamp, and Duration, total number of shifts), 
    and a summary of Gaze shifts (Direction, Timestamp, Count of total gaze shifts)

    Attributes:
        fname (str):           A path to the raw video file to be processed.
        PID (str):             A participant ID to associate with the data.
        trial_id (int):        An ID number for each trial to associate with the date.
        settings (dict):       A dictonary objecte of settings generated from settings_util module.
        frame_dir (str):       A path to a directory where images of each frame will be stored, if desired. 
        multiple_id (bool):    Boolean indicating if multiple ID procedures should be followed. Default is to use largest face. 
        face_id (int):         Intiger of the ID for the desired face. ID number can be found from the multiple_ID procedures. 
        output_level (str):    Level of output for etdset provided (basic, or intermediate, or full).
        preprocessing (bool):  boolean to identify if video should be preprocessed (recomended to ensure correct meta data).
        overwrite_video(bool): boolean to identify whether preprocessing should overwrite previosuly preprocessed videos. 
        processed_fname (str): path to where preprocessed videos should be saved.
        format (str):          string of the default format for preprocessed videos.
        set_fps (str):         set the fps if known
        start_time (int):      The time in seconds at which the experimental trial starts in the video
        stop_time (int):       The time in seconds when the experimental trial ends in the video
        video_length (int):    The length of the experimental trial in seconds. Can be used either alone or with start/stop_time
                               to set correct video length.
        apply_lkv (bool):      confirms whether a last know value procedure should be applied to the etdset to remove blinks etc. 

    """

    def __init__(self, fname, PID=None, trial_id=None, settings=None, frame_dir=None, multiple_id=True,
                 face_id=None, output_level="intermediate", preprocessing=True, overwrite_video=False, processed_fname=None,
                 format=".mp4", set_fps=None, start_time=None, stop_time=None, video_length=None, apply_lkv=True):

        """Inits processing of video files to estimate gaze orientation. Data is saved in a dataframe called etdset where each row corresponds to a single frame"""

        # adds required varibles to self   
        self.output_level = output_level
        self.aratios = None
        self.fname = fname

        # if settings are not provided will generate from the settings utils module
        if settings is None:
            self.settings = settings_utils.assign_settings()
        else:
            self.settings = settings

        # Gathers meta data from raw video. Sets fps manually if provided. 
        if set_fps is not None:
            self.meta = video_utils.get_meta_data(self.fname)
            self.meta["fps"] = set_fps
        else:
            self.meta = video_utils.get_meta_data(self.fname)   


        # Preprocesses the video (if desired) with default format as .mp4. Also trims video to desired length based on start/stop/length inputs. 
        if preprocessing:
            self.fname = video_utils.preprocess_video(
                self.fname, meta=self.meta, overwrite_video=overwrite_video,
                processed_fname=processed_fname, format=format, set_fps=set_fps, start_time=start_time, stop_time=stop_time, video_length=video_length)

        #Processes the inpuput video by seperating the files to individual frames and processing them individually
        self.etdset, self.aratios = video_utils.video_processing(self.fname, self.meta, settings=self.settings,
                                            frame_dir=frame_dir, multiple_id=multiple_id, face_id=face_id, output_level=output_level, apply_lkv=apply_lkv, require_aratios=True)

        # Adds PID and trial data to etdset if provided   
        if PID is not None:
            self.etdset["PID"] = PID
        if trial_id is not None:
            self.etdset["trialID"] = trial_id

    
    def __call__(self, output='etdset'):
        

        """Returns the required datasets defined by outut argument. Default=etdset
        """                     
        # Default outpu - full eye tracking dataset
        if output == 'etdset':
            return self.etdset
        #output of gazeshift data    
        elif output == 'gazeshift':
            return self.gazeshift_output
        #output of firstlook data
        elif output == 'firstlook':
            return self.firstlook_output
        #output of proportion data
        elif output == 'proportion':
            return self.proportion_output 
    
       
    

    
    def add_timestamp(self,
                      h_event_start=None,
                      h_location_key=None,
                      h_movement_key=None,
                      v_event_start=None,
                      v_location_key=None,
                      v_movement_key=None,
                      ):
        """Calculates the targets location based on the timestamps and appends a key for each frame to the data frame.

        Args:
            h_event_start (list): timepoint corrosponding to a new target location event for horizontal movements
            h_location_key (list): location key that corresponds to the target horizontal location. Length must match h_event_start.
            h_movement_key (list): key to mark if target is moving horizontally during the target event. Length must match h_event_start.
            v_event_start (list):  timepoint corrosponding to a new target location event for vertically movements
            v_location_key (list): location key that corresponds to the target vertically location. Length must match v_event_start.
            v_movement_key (list): key to mark if target is moving vertically during the target event. Length must match v_event_start.

        Returns:
            self: Updates the current self.etdset with appended timestamps
        """

        self.etdset = post_utils.add_timestamp(self.etdset, 
                    h_event_start = h_event_start,
                    v_event_start = v_event_start,
                    h_location_key = h_location_key,
                    h_movement_key = h_movement_key,
                    v_location_key = v_location_key,
                    v_movement_key = v_movement_key)
        
        return self

    def add_scoring(self, left_eye=None, right_eye=None, overall_scoring=None, apply_lkv=True):

        """Adds alternative scoring to the dataset

            Adds alternative scoring to the data set, for example, from a manual scorer. It will also apply a LKV procedure to the data.
            The data provided must be the same length as the etdset. 

        Args:
            left_eye (list): list of values for each frame from left eye.
            right_eye (list): list of values for each frame from the right_eye.
            overall_scoring (list): list of values for each frame for overall scoring.
            apply_lkv (bool): boolean to confirm if a LKV process should be applied.

        Return:
            self: Updates the current self.etdset with appended timestamps
        """

        self.etdset = post_utils.add_scoring(self.etdset, left_eye=left_eye, right_eye=right_eye, overall_scoring=overall_scoring, output_level=self.output_level, apply_lkv=apply_lkv)

        return self


    def run_plots(
        self,
        settings=None,
        PID=None,
        do_phcoords=None,
        do_hdir=None,
        do_aratio=None,
        aratios=None,
    ):

        if settings is None:
            settings = settings_utils.assign_settings()

        proc = settings["proc"]
        gaze = settings["gaze"]
        plots = settings["plots"]

        if do_phcoords is not None:
            if PID is not None:
                plotfname = f"{do_phcoords}/{PID}_phcoords.png"
            else:
                PID = ""
                plotfname = f"{do_phcoords}_phcoords.png"

            plot_utils.plot_phcoords(
                self.etdset, PID, gaze["elist"], plots, plotfname)

        if do_hdir is not None:
            if PID is not None:
                plotfname = f"{do_hdir}/{PID}_hdir.png"
            else:
                PID = ""
                plotfname = f"{do_hdir}_hdir.png"

            plot_utils.plot_hdir(
                self.etdset, PID, gaze["elist"], plots, plotfname)

        if do_aratio is not None:

            if self.aratios is None:
                print("aratios variable not entered.")

            if PID is not None:
                plotfname = f"{do_aratio}/{PID}_aratio.png"
            else:
                PID = ""
                plotfname = f"{do_aratio}_aratio.png"

            plot_utils.plot_aratio(self.etdset, self.aratios, PID,
                                   gaze["elist"], plots, plotfname)

    def save_csv(self, fname):

        """Save etdset to a csv at fname (str) path
        """
        self.etdset.to_csv(fname)

    def pp_proportion(self,  eye='overall', specify_measure=None):
        
        """Calculates proportion of time spent looking at each location

            Calculates proportion of time spent looking at each location. The default is to use the overall data, but can be run on a single eye. 
            It is also possible to specify whether Time, Proportion, or Both are provided

        Args:
            eye (str): string defining which eye to use. Default = overall
            specify_measure (str): string defining which measure to output ('time' or 'proportion'). None=both

        Return:
            self: crates new datset named self.proportion_output
        """

        self.proportion_output = post_utils.proportions(self.etdset, eye=eye, specify_measure=specify_measure)

        return self

    def pp_first_look(self, eye='overall', fix_buffer=5, n_shifts=False):

        """Ouputs the first look direction

            Identifies the first look direction and provides the timestamp and duration (plus optionally number of shifts overall).
            The first look is defined by the first direction shift that is consistant accross a number of frames defined by the fix_buffer variable.

        Args:
            eye (str): string defining which eye to use. Default = overall
            fix_buffer (int): Defines the min number of frames required to define a look
            n_shifts (bool): boolean to identify if number of shifts should be reported.

        Return:
            self: creates new datset named self.firstlook_output
        """

        self.firstlook_output = post_utils.firstlook(self.etdset, eye=eye, fix_buffer=fix_buffer, n_shifts=n_shifts)

        return self

    def pp_gaze_shift(self, eye='overall', fix_buffer=5):

        """Ouputs a dataframe of each gaze shift accross the course of the video analysis

            Identifies each change in gaze orientation and provides the timestamp, duration, and current number of gaze shifts.
            A look is defined by the a direction shift that is consistant accross a number of frames defined by the fix_buffer variable.

        Args:
            eye (str): string defining which eye to use. Default = overall
            fix_buffer (int): Defines the min number of frames required to define a look
           

        Return:
            self: creates new datset named self.gazeshift_output
        """


        self.gazeshift_output = post_utils.gaze_shift(self.etdset, eye=eye, fix_buffer=fix_buffer)

        return self

   