# alternate proceess for extracting frames for comparison on BRM paper
def extract_video_frames_alt(vfname, frame_dir=None, PID=None):

    video = []

    try:
        probe = ffmpeg.probe(vfname)
    except ffmpeg.Error as e:
        print("stdout:", e.stdout.decode("utf8"))
        print("stderr:", e.stderr.decode("utf8"))
        raise e

    video_stream = next(
        (stream for stream in probe["streams"]
            if stream["codec_type"] == "video"), None
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])

    meta = get_meta_data(vfname)

    timestamps = meta["time_stamps"][:-1]
    print(len(timestamps))

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
    processed_dir=None,
    reprocess_video=False,
):

    if overwrite_video:

        fname = vfname

    else:

        if processed_dir is None:
            processed_dir = f"{os.path.dirname(vfname)}/processed/"

        utils.create_folder(processed_dir)
        fname = os.path.splitext(os.path.basename(vfname))

        fname = f"{fname[0]}_processed{fname[1]}"

        fname = f"{processed_dir}/{fname}"

    while True:

        # Check if file exists and should be reprocessed
        if os.path.isfile(fname) and reprocess_video == False:
            print(
                f"File {fname} already existis and video preprocessing will be skipped."
            )
            break

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
        os.system(command_video)
        break

    return fname


def process_images(

    vfname,
    settings=None,
    save_csv=None,
    PID=None,
    output_level="intermediate",
    h_location=None,
    h_key=None,
    h_moving=None,
    v_location=None,
    v_key=None,
    v_moving=None,
    scoring_left=None,
    scoring_right=None,
    scoring_overall=None,
    apply_lkv=False,
    plot_face_lmarks=None,
    plot_face_lmarks_all=None,
    plot_iris_lmarks=None,
    plot_iris_lmarks_all=None,
    do_phcoords=None,
    do_hdir=None,
    do_aratio=None,
):

    if settings is None:
        settings = settings_utils.assign_settings()

    # Set up bools and constant values for processing
    include_scoring = False
    # TO DO - SET UP META AS CONSTANT VARIBALE HERE

    # # Extract the frames into a numpy array
    frames = extract_image_frames(vfname)

    # Process the frames to detect landmarks
    etdset = image_utils.process_frames(
        frames,
        settings,
        plot_face_lmarks=plot_face_lmarks,
        plot_iris_lmarks=plot_iris_lmarks,
        plot_face_lmarks_all=plot_face_lmarks_all,
        plot_iris_lmarks_all=plot_iris_lmarks_all,
    )

    # # add timestamps to the data set
    # etdset["frame_timestamp"] = tstamps

    # Process the lanmarks to make location decisions
    etdset = image_utils.process_lmarks(
        etdset, settings, do_phcoords=do_phcoords, do_hdir=do_hdir, do_aratio=do_aratio
    )

    # checks if horizontal timestamps are supplied and adds to the data set
    if h_location is not None or h_key is not None:

        if h_key is None:
            print(
                "Horizontal location key is was not provided. Horizontal scores will not be added"
            )
        elif h_location is None:
            print(
                "Horizontal location timepoints were not provided. Horizontal scores will not be added"
            )
        else:
            etdset = add_timestamp(
                etdset,
                h_event_start=h_location,
                h_location_key=h_key,
                h_movement_key=h_moving,
            )

    # checks if vertical timestamps are supplied and adds to the data set
    if v_location is not None or v_key is not None:

        if v_key is None:
            print(
                "Vertical location key is was not provided. Horizontal scores will not be added"
            )
        elif v_location is None:
            print(
                "Vertical location timepoints were not provided. Horizontal scores will not be added"
            )
        else:
            etdset = add_timestamp(
                etdset,
                v_event_start=v_location,
                v_location_key=v_key,
                v_movement_key=v_moving,
            )

    # Checks if scoring has been supplied and adds them to the data set. Also changes include_scoring variable to True
    if (
        scoring_left is not None
        or scoring_right is not None
        or scoring_overall is not None
    ):
        include_scoring = True
        etdset = add_scoring(
            etdset,
            left_eye=scoring_left,
            right_eye=scoring_right,
            overall_scoring=scoring_overall,
        )

    # Prepares the output by cutting excess columns and adding in PID
    etdset = prepare_output(etdset, output_level=output_level, PID=PID)

    if apply_lkv:
        # Applies the LKV processing to the data set to remove blinks and other lost data.
        # Include scoring indicates if lkv should also be applied for scoring values
        etdset = lkv_process(etdset, include_scoring=include_scoring)

    # Finally saves the data set to csv if requested
    if save_csv is not None:
        etdset.to_csv(save_csv)

    # # AF run plot_visuals function if do_visual is True. Maps automatic scoring onto face image
    # if do_visual  is not None:
    #     npnts, flmarksinds, flmarkscolours = image_utils.set_lmarksvars(
    #         proc["shape_predictor"])
    #     plot_utils.plot_visuals(folders, info, gaze, flmarksinds, plots)

    return etdset

# Add scoring to if multiple participants. 
def add_scoring(self, PID, left_eye=None, right_eye=None, overall_scoring=None):

        
        etdset = []

        temp_etdset = self.etdset[self.etdset["PID"] == PID].copy()
        

        if left_eye is not None:
            if len(left_eye) != len(temp_etdset[temp_etdset["PID"] == PID]):
                print(
                    "Length of left eye data does not match the length of the data set and will be skipped"
                )
                left_eye = None
            else:
                self.etdset.loc[self.etdset['PID'] == PID, "slgazeraw"] = left_eye
                
                # left_eye.copy()

        if right_eye is not None:
            if len(right_eye) != len(temp_etdset):
                print(
                    "Length of right eye data does not match the length of the data set and will be skipped"
                )
                right_eye = None
            else:
                # temp_etdset["srgazeraw"] = right_eye.copy()
                self.etdset.loc[self.etdset['PID'] == PID, "srgazeraw"] = right_eye

        if overall_scoring is not None:
            if len(overall_scoring) != len(temp_etdset):
                print(
                    "Length of overall_scoring data does not match the length of the data set and will be skipped"
                )
                overall_scoring = None
            else:
                # temp_etdset["sgazeraw"] = overall_scoring.copy()
                self.etdset.loc[self.etdset['PID'] == PID, "sgazeraw"] = overall_scoring

        return self

def extract_image_frames(img_dir):

        """Reads image files from a provided directory and saves them to an array for analysis
        """
              
        img_paths = glob.glob(f"{img_dir}/*")
        img_paths = sorted(img_paths)

        frames = []

        for i in img_paths:
            frames.append(cv2.imread(i))

        return frames