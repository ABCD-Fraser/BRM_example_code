from concurrent.futures import process
import os
import pandas as pd
import subprocess
import cv2
import ffmpeg
import numpy as np

from GazeScorer import settings_utils
from GazeScorer import image_utils
from GazeScorer import post_utils
from GazeScorer import utils



def preprocess_video(
        vfname,
        meta = None,
        overwrite_video=False,
        processed_fname=None,
        format=".mp4",
        set_fps=None,
        start_time=None,
        stop_time=None,
        video_length=None,
        require_meta=None,
        alt_processing=False
    ):
        """
        Calculates approprate start and stop times and trims and rencodes the videos for future processing.


        Args:
            vfname (str): String of path to raw video file
            meta (dict): dictionary of meta data if already created
            overwrite_video (bool): sets if raw video should be overwritten. Default does not overwrite files
            processed_fname (str): path of location to save processed video
            format (str): Format to convert preprocessed videos to. Default mp4. 
            set_fps (float): Specific fps for processing videos
            start_time (int): desired start time of video in ms (optional)
            stop_time (int): desired stop time of video in ms (optional)
            video_length (int): length of video
            require_meta (bool): Boolean to identify if output of meta data is required.
            alt_processing (bool): Boolean to id if older processing should be used.  


        Returns:
            processed_vfname (str): path to the processed video file
            meta  (dict): optional - dictionary of video meta data


        """
        try:
            
            if meta is None and set_fps is None:
                meta = get_meta_data(vfname)
                if require_meta is None:
                    require_meta = True
            elif meta is None and set_fps is not None:
                meta = get_meta_data(vfname)
                meta['fps'] = set_fps
                if require_meta is None:
                    require_meta = True
                
            if processed_fname is None:
                fname = f"{os.path.splitext(vfname)[0]}_processed{format}"
            else:
                if os.path.isfile(processed_fname) and (
                    overwrite_video == False or overwrite_video == 0):       
                    # Check if file exists and should be reprocessed
                    print(
                        f"File {processed_fname} already existis and video preprocessing will be skipped."
                    )
                    
                    if require_meta:
                        return processed_fname, meta
                    else:
                        return processed_fname 

                elif os.path.isfile(processed_fname) and (
                    overwrite_video == True or overwrite_video == 1):
                    print(
                    f"File {processed_fname} already existis. Video will be overwritten"
                    )
                    fname = processed_fname
                else:
                    fname = processed_fname
                
                    
                
            
                
            # Start and stop times are provided, but not video length.
            if start_time is not None and stop_time is not None and video_length is None:

                print(f"Cutting video betweeen {start_time} and {stop_time}")
                if start_time < 0 or stop_time > meta['duration']:
                        raise Exception(f"Start time and/or stop time was outside of the video parameters.")
                # makes the command to feed into ffmpeg
                command_video = f"ffmpeg -ss {(start_time * 1000)} -t {(stop_time * 1000)} -i {vfname}  -y  {fname}"
                # logger.info(f'Processig video {filename_video_output}')
                subprocess.run(command_video, shell=True)
                # break

            # Start time and video length provided - stop time is caluclated
            elif start_time is not None and video_length is not None and stop_time is None:
                start_time = start_time
                stop_time = start_time + video_length

                print(f"Cutting video betweeen {start_time} and {stop_time}")
                if start_time < 0 or stop_time > meta['duration']:
                    raise ValueError(f"Start time and/or stop time was outside of the video parameters.")

                # makes the command to feed into ffmpeg
                command_video = f"ffmpeg -ss {(start_time * 1000)} -t {(stop_time * 1000)} -i {vfname}  -y  {fname}"
                # logger.info(f'Processig video {filename_video_output}')
                subprocess.run(command_video, shell=True)
                # break

            # Stop time and video length provded - start time is calculated
            elif stop_time is not None and video_length is not None and start_time is None:
                stop_time = stop_time
                start_time = stop_time - video_length

                print(f"Cutting video betweeen {start_time} and {stop_time}")

                if start_time < 0 or stop_time > meta['duration']:
                    raise Exception(f"Start time and/or stop time was outside of the video parameters.")

                    
                # makes the command to feed into ffmpeg
                command_video = f"ffmpeg -ss {(start_time * 1000)} -t {(stop_time * 1000)} -i {vfname}  -y  {fname}"
                # logger.info(f'Processig video {filename_video_output}')
                subprocess.run(command_video, shell=True)
                # break

            # Only video length is provided
            elif video_length is not None and start_time is None and stop_time is None:

                # +ve value provided, stop point caluculated
                if video_length > 0:
                    start_time = 0
                    stop_time = start_time + video_length

                    print(
                        f"Cutting video betweeen {start_time} and {stop_time}")
                    
                    if start_time < 0 or stop_time > meta['duration']:
                        raise Exception(f"Start time and/or stop time was outside of the video parameters.")

                    # makes the command to feed into ffmpeg
                    command_video = f"ffmpeg -fflags +genpts -ss {start_time} -t {stop_time} -i {vfname} -r {meta['fps']} -qscale 0 -y {fname}"
                    # logger.info(f'Processig video {filename_video_output}')
                    subprocess.run(command_video, shell=True, check=True)
                    # break

                # -ve value provided, start point calculated
                else:

                    stop_time = meta['duration']
                    print(f"duration = {meta['duration']}")

                    video_length = abs(video_length)
                    print(f"video_length = {video_length}")

                    start_time = stop_time - video_length

                    print(f"start time = {start_time}")

                    print(
                        f"Cutting video betweeen {start_time} and {stop_time}")
                    
                    if start_time < 0:
                        raise ValueError(f"Start time ({start_time}) was less than 0")
                    elif stop_time > meta['duration']:
                        raise ValueError(f"Stop top ({stop_time}) was beyond the length of the video file ({meta['duration']}")
                    
                    # makes the command to feed into ffmpeg
                    command_video = f"ffmpeg -fflags +genpts -ss {start_time} -t {stop_time} -i {vfname} -r {meta['fps']} -qscale 0 -y {fname}"
                    
                    if alt_processing:
                        command_video = f"ffmpeg -hide_banner -loglevel error -ss {start_time}  -i {vfname} -lossless 1 -y  {fname}"
                    # print(command_video)
                    # logger.info(f'Processig video {filename_video_output}')
                    subprocess.run(command_video, shell=True, check=True)
                    # break

            # Incorrect combination of variables provided.
            else:
                print(f"Reencoding video")
                # makes the command to feed into ffmpeg
                command_video = f"ffmpeg -fflags +genpts -i {vfname} -r {meta['fps']} -qscale 0 -y {fname}"
                # logger.info(f'Processig video {filename_video_output}')
                subprocess.run(command_video, shell=True)
                # break

            meta['stop_time'] = stop_time  

            
            if require_meta:
                return fname, meta
            else:
                return fname      
                
        except Exception as error:
            raise RuntimeError(f'Unable to process video file: {vfname}. Please check that the file path is correct.\n It is also possible that the file is corrupt')

        
            

           
        
        

def get_meta_data(vfname):

    """Functio used to prob raw video files and return dictionary of core meta data
    """
    
    try:

        frames = subprocess.check_output(
            f"ffprobe -hide_banner -loglevel error {vfname} -show_frames -show_entries frame=pkt_pts_time -of csv=p=0",
            shell=True,
        )

        
        tstamps = frames.decode().splitlines()
        duration = float(tstamps[-1])
        frame_count = len(tstamps)

        fps = frame_count / duration
        
        meta = {
            "fps": fps,
            "duration": duration,
            "frame_count": frame_count,
            "time_stamps": tstamps
        }

        return meta

    except Exception as error: 
        raise RuntimeError(f'Unable to probe video file: {vfname}. Please check that the file path is correct.\n It is also possible that the file is corrupt')
    

def video_processing(
        vfname,
        meta=None,
        settings=None,
        PID=None,
        frame_dir=None,
        alt_frames=False,
        output_level="basic",
        apply_lkv=True,
        plot_face_lmarks=None,
        plot_face_lmarks_all=None,
        plot_iris_lmarks=None,
        plot_iris_lmarks_all=None,
        multiple_id=True,
        face_id=None,
        require_aratios=False
    ):

    """Gather gaze orientation data from video file

        This functuon extracts the etdset from a video once video has been preprocessed (if required). 
        Dependent on the output_level a variety of data will be provided as part of the etdset function. 

    Args:
        vfname (str):           A path to the raw video file to be processed.
        meta (dict): dictionary of meta data if already created
        settings (dict):       A dictonary objecte of settings generated from settings_util module.
        PID (str):             A participant ID to associate with the data.
        frame_dir (str):       A path to a directory where images of each frame will be stored, if desired. 
        output_level (str):    Level of output for etdset provided (basic, or intermediate, or full).
        apply_lkv (bool):      confirms whether a last know value procedure should be applied to the etdset to remove blinks etc.
        plot_face_lmarks (bool): creates an image of the first frame with the face plotted
        plot_face_lmarks_all (bool):  creates an image of all frames with the face plotted
        plot_iris_lmarks (bool): creates cropped image of the plotted iris for the first frame
        plot_iris_lmarks_all (bool):  creates cropped image of the plotted iris for all frames
        multiple_id (bool):    Boolean indicating if multiple ID procedures should be followed. Default is to use largest face. 
        face_id (int):         Intiger of the ID for the desired face. ID number can be found from the multiple_ID procedures. 
        require_aratios (bool): Boolean to identify if aratios are required outputs

    Returns:
        etdset (datframe): data frame of face landmarks and gaze orientation data (output dependent on level provided).
        aratios (dataframe): dataframe of aratio values 
        


    """

    if settings is None:
        settings = settings_utils.assign_settings()

    
    # # Extract the frames into a numpy array
    print('Extracting video frames')
    if alt_frames == True:
        frames, tstamps = extract_video_frames_alt(
            vfname, frame_dir, PID)
    else:
        frames, tstamps = extract_video_frames(vfname, meta=meta, frame_dir=frame_dir, PID=PID)

    # Process the frames to detect landmarks
    print('processing video frames')
    etdset = image_utils.process_frames(
        frames,
        settings,
        face_id=face_id,
        multiple_id=multiple_id,
        plot_face_lmarks=plot_face_lmarks,
        plot_iris_lmarks=plot_iris_lmarks,
        plot_face_lmarks_all=plot_face_lmarks_all,
        plot_iris_lmarks_all=plot_iris_lmarks_all,
    )

    # add timestamps to the data set
    etdset["frame_timestamp"] = tstamps

    # Process the landmarks to make location decisions
    etdset, aratios = image_utils.process_lmarks(
        etdset, settings)

    print('Preparing data output')
    # reassign column names for readability
    etdset = post_utils.rename_cols(etdset)

    if apply_lkv:
        # Applies the LKV processing to the data set to remove blinks and other lost data.
        # Include scoring indicates if lkv should also be applied for scoring values
        etdset = post_utils.lkv_process(etdset, settings=settings, output_level='full')


    # Prepares the output by selecting the correct columns depending on the output level
    etdset = post_utils.prepare_output(
        etdset, output_level=output_level, PID=PID)

    if require_aratios:
        return etdset, aratios
    else:
        return etdset


def extract_video_frames(vfname, meta=None, frame_dir=None, PID=None, alt_processing=False):
    """
    Extract frames and creates a corrosponding time stamp with each frame

    Args:
        folders (dict): arguments with settings for creating input/output path
        video (dict): settings for processing videos
        PID (int): Participant ID number
        cut_video (str): String of path to video trimmed to trial length

    Returns:
        frame_list (list): list of frame image paths
        cut_frames_tstamp (list): list of frame timestamps

    """

    if meta is not None:
        meta = get_meta_data(vfname)
        meta['stop_time'] = meta['duration']

    count = 1
    success = True

    vidcap = cv2.VideoCapture(vfname)

    frames = []
    tstamp_list = []

    success, image = vidcap.read()

    if alt_processing: 
        fps = meta['fps']
    else:
        fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    tstamp = 0 
    while success:
        frames.append(image)
        tstamp_list.append(tstamp)

        if frame_dir is not None:
            if PID is not None:
                cv2.imwrite(
                    f"{frame_dir}/{PID}_frame_{count:04}.png", image)
            else:
                cv2.imwrite(f"{frame_dir}/frame_{count:04}.png", image)

        success, image = vidcap.read()
        if alt_processing: 
            fps = meta['fps']
        else:
            fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count += 1
        tstamp = float(frame_count)/fps



        count += 1

    print(f'Processsing {len(frames)} frames')
    return frames, tstamp_list

# alternate proceess for extracting frames for comparison on BRM paper
def extract_video_frames_alt(vfname, meta=None, frame_dir=None, PID=None):

    video = []

    try:
        probe = ffmpeg.probe(vfname)
    except Exception as e:
        raise e

    video_stream = next(
        (stream for stream in probe["streams"]
            if stream["codec_type"] == "video"), None
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])

    if meta is None:
        meta = get_meta_data(vfname)

    timestamps = meta["time_stamps"][:-1]
    # print(len(timestamps))

    # print(len(timestamps))
    missed_frame = 0

    for i, time in enumerate(timestamps):

        # print(i)

        count = i + 1

        out, _ = (
            ffmpeg.input(vfname, ss=time)
            .output("pipe:", vframes=1, format="rawvideo", pix_fmt="bgr24")
            .run(capture_stdout=True)
        )

        try:
            out = np.frombuffer(out, np.uint8).reshape([height, width, 3])

            video.append(out)

            if frame_dir is not None:
                if PID is not None:
                    cv2.imwrite(
                        f"{frame_dir}/{PID}_frame_{count:04}.png", out)
                else:
                    cv2.imwrite(f"{frame_dir}/frame_{count:04}.png", out)

        except:
            timestamps = timestamps[:-1]
            missed_frame += 1
            break

    # video = np.array(video)
    return video, timestamps

# alternate prerpocessing for comparison on BRM paper
def preprocess_video_alt(
    vfname,
    video_length,
    overwrite_video=False,
    processed_fname=None,

):


    if processed_fname is None:
        fname = f"{os.path.splitext(vfname)[0]}_processed{format}"
    else:
        if os.path.isfile(processed_fname) and (
            overwrite_video == False or overwrite_video == 0):       
            # Check if file exists and should be reprocessed
            print(
                f"File {processed_fname} already existis and video preprocessing will be skipped."
            )
            return processed_fname
            

        elif os.path.isfile(processed_fname) and (
            overwrite_video == True or overwrite_video == 1):
            print(
            f"File {processed_fname} already existis. Video will be overwritten"
            )
            fname = processed_fname
        else:
            fname = processed_fname
                


    while True:


        # probe the video to get frame info

        command_frame_outline = f"ffprobe -i {vfname}  -show_frames -show_entries frame=pkt_pts_time -of csv=p=0"
        frames_string = str.splitlines(
            subprocess.check_output(
                command_frame_outline, shell=True, universal_newlines="\n"
            )
        )
        integer_map = map(float, frames_string)
        frames_timestamp = list(integer_map)
        duration = frames_timestamp[-1]
        del frames_timestamp[-1]

        # cut vidoe to correct length
        cut_time = duration - video_length

        # makes the command to feed into ffmpeg
        
        command_video = (
            f"ffmpeg -ss {cut_time} -i {vfname}  -y  {fname} >/dev/null 2>&1"
        )
        
        

        print(f"Processig video {vfname}")
        print(command_video)
        os.system(command_video)
        break

    return fname