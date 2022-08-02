import logging
from datetime import datetime

from GazeScorer import utils


def assign_logger(path):
    """
    Assigns settings for logging

    Args:
        path (str): path to where the log files should be savec
    Returns:
        none
    """

    _, logstatus = utils.create_folder(path)

    # set up logger
    logtime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    logfile = f"{path}{logtime} _test.log"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")

    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.WARNING)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    if not logstatus:
        logger.error("Could not create output folder")


def assign_settings():
    
    """Assigns settings used at different stages of the pipeline

    Args:
        none
    Returns:
        info (dict): settings for input data source
        folders (dict): settings for output folders
        plots (dict): settings for plots
        gaze (dict): settings for gaze scoring and labelling
        proc (dict): settings for processing of the frames and raw landmarks output
    """

    # plot settings
    # - face plots
    plots = {}
    plots["fcolour"] = "w"
    plots["flinewdt"] = 2
    # - eye plots
    plots["ecolours"] = ["b", "r", "g"]  # Left: bLue,   Right: Red
    plots["emarkers"] = ["o", "*", "d"]  # Left: circLe. right: staR
    plots["emultipliers"] = [-0.1, 0.1]  # offset multipliers for plotting
    plots["enames"] = ["left", "right", "overall"]
    plots["elinewdt"] = 2
    plots["etstamps_offset"] = 0.4
    # plots selection
    plots["do_lmarks"] = False
    plots["do_whole"] = False
    plots["do_iris"] = False
    plots["do_list"] = False  # if True save all plots in one folder
    plots["do_annotations"] = True
    plots["do_phcoords"] = True
    plots["do_hdir"] = True
    plots["do_aratio"] = True
    # AF added setting to select whether to do visual annotation, scoring plots and cohen kappa plots
    plots["do_visual"] = True  # plot visual annotations on face images
    plots["do_scoring"] = True  # add manual scoring for visual annotations
    plots["do_splots"] = True  # do manual vs tstamps scoring plots
    plots["do_aplots"] = True  # do automatic vs tstamps scoring plots
    plots["do_ckplots"] = True  # do Cohen-Kappa plots
    # plots dpi
    plots["dpi"] = 40

    # folder settings
    folders = {}
    folders["video"] = "output/videos/"
    folders["frames"] = "output/frames/"
    folders["proc"] = "output/processed/"
    folders["output"] = "output/face_img/"
    folders["plots"] = "output/plots/"
    folders["irisinfo"] = "input/annotations/"
    # AF added scorer paths and list of socrers
    folders["spath"] = "input/scoring/scorers/"
    folders["scorers"] = ["scorer01"]  # , "scoring/scorers/scorer02"]
    folders["tstamps"] = "output/tstamps/"
    folders["face"] = "output/" + folders["output"] + "face/"
    folders["eyes"] = "output/" + folders["output"] + "eyes/"
    # AF Added in folder name for annotated frames.
    folders["aframes"] = "output/" + folders["output"] + "aframes/"
    folders["annotate"] = "output/annotations/"
    folders["logs"] = "output/logs/"
    # AF folders for video extract
    folders["video_in"] = "input/video_input/"
    folders["video_out"] = "output/video_output/"
    folders["stim_stamp"] = "input/stim_stamp/"

    info = {}
    # info["study"]        = study
    info["imgextension"] = ".png"
    info["imgprefix"] = "output_"
    info["fname"] = "respondent_info.xlsx"
    info["sheet"] = "info"

    # Video extract settings
    video = {}
    # , '_task-ym1w.csv', '_task-n3gg.csv']
    video["files"] = ["_task-6ymf.csv"]
    video["overwrite"] = [True]
    video["vid_ext"] = "_cut.webm"
    video["frame_ext"] = "_output_%05d.png"
    video["tstamp_ext"] = "_timestamp.csv"
    video["countdown"] = 6.07
    video["vlength"] = 20.48 - video["countdown"]

    # eyes/looks setttings
    gaze = {}
    gaze["elist"] = ["left", "right"]
    gaze["elist_short"] = ['l', 'r']
    gaze["hdirlist"] = ["RIGHT", "MIDDLE", "LEFT"]
    gaze["vdirlist"] = ["UP", "MIDDLE", "DOWN"]

    gaze["hdirlabels"] = ["RIGHT", "MIDDLE", "LEFT", "BLINK", "OUTSIDE", "UNKNOWN"]
    gaze["hdirvals"] = [-1, 0, 1, 99, 3, 4]

    gaze["hdirs"] = {}
    gaze["hdirs"]["RIGHT"] = -1
    gaze["hdirs"]["MIDDLE"] = 0
    gaze["hdirs"]["LEFT"] = 1
    gaze["hdirs"]["BLINK"] = 99
    gaze["hdirs"]["OUTSIDE"] = 3
    gaze["hdirs"]["UNKNOWN"] = 4

    proc = {}
    proc["shape_predictor"] = "shape_predictor_68_face_landmarks.dat"
    # proc["shape_predictor"] = "shape_predictor_5_face_landmarks.dat"
    proc["ehbufratio"] = 0.0
    proc["do_rounding"] = True
    proc["do_filtering"] = True
    proc["nflt"] = 9
    proc["n4fxt"] = 4
    proc["do_adjustment"] = True
    proc["nadjustmentpnts"] = 15  # half a second for a 30fps video
    proc["do_blinks"] = True
    proc["emidbuf"] = "ratio"  # "std"
    proc["emidwdtratio"] = 0.05
    # multiplication factor for std around the middle
    proc["emidstdratio"] = 1.5

    proc["columns2round"] = []
    proc["columns2filter"] = []
    for eye in gaze["elist"]:
        varlist = [
            "e" + eye[0] + "xmidraw",
            "e" + eye[0] + "ymidraw",
            "i" + eye[0] + "x",
            "i" + eye[0] + "y",
        ]
        [proc["columns2round"].append(var) for var in varlist]
        [proc["columns2filter"].append(var) for var in varlist]

    proc["columns2filter"].append("aratioavg")
    proc["percent_error_limit"] = 30  # percent of frames to count as error

    settings = {
        "info": info,
        "folders": folders,
        "plots": plots,
        "gaze": gaze,
        "proc": proc,
        "video": video,
    }

    return settings
