# import cv2
# import pandas as pd
# import numpy as np
# import ast
# import logging


# # def read_video(fname):
#     """Read a video file.

#     Args:
#         fname (string): input filename

#     Returns:
#         vid: handle to open video
#         nframes (int): number of frames in the video
#         fps (int): frames per second

#     """

#     vid = cv2.VideoCapture(fname)

#     if vid.get(cv2.CAP_PROP_FRAME_COUNT) == 0:
#         print('Failed to open ', videofolder + vidfname + vidextension)
#         nframes = None
#         fps = None
#     else:
#         print('Successfully read ', fname)
#         nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = int(vid.get(cv2.CAP_PROP_FPS))
#         print('- number of frames:  ', nframes)
#         print('- frames per second: ', fps)
#         print('- video duration:    ', (nframes / fps), ' seconds')
#         print('- file position:     ', vid.get(cv2.CAP_PROP_POS_MSEC), 'msec')

#     return vid, nframes, fps


# def read_scoring(fname):
#     """Read a scoring file.

#     Codes for manual scoring - gaze R/L are scorer's R/L so they need swapping
#     via output - 0:"M", 1:"R", 2:"L", 3:"B", 4:"O", 5:"U"

#     Args:
#         fname (string): input filename
#     Returns:
#         scoring (dataframe): processed manual scoring

#     """

#     scoring = pd.read_csv(fname, index_col=0)
#     scoring.index = np.arange(1, len(scoring) + 1)

#     # recode manual scoring to match automatic processing
#     # scoring.replace(1,  -1,  inplace=True) # scorer  R (1) -> gaze L (-1)
#     # scoring.replace(2,   1,  inplace=True) # scorer  L (2) -> gaze R (1)
#     # recode manual scoring to match automatic processing
#     # scoring.replace(1,  -1,  inplace=True) # scorer  R (1) -> gaze L (1)
#     scoring.replace(2, -1, inplace=True)  # scorer  L (2) -> gaze R (-1)
#     scoring.replace(3, 2, inplace=True)   # blink   B (3) -> 2
#     scoring.replace(4, 3, inplace=True)   # outside O (4) -> 3
#     scoring.replace(5, 4, inplace=True)   # unknown U (5) -> 4

#     return scoring


# def process_scoring(dset, respondent, prefix, fext):
#     """Process a scoring dataframe.

#     Args:
#         dset (dataframe): dataset with read scoring
#         respondent (string): respondent id that is also the name for the frames folder (as in via scoring column)
#         prefix (string): prefix for the frames file names
#         fext (string): extension for the frame files
#     Returns:
#         dsetproc (dataframe): dataset wit processed information
#         status (bool): flag for the status of the processing

#     """
#     # AFinitiate logger
#     logger = logging.getLogger(__name__)

#     # default values
#     status = True
#     reye = None
#     leye = None
#     overall = None

#     # dataframe to keep processed scoring
#     dsetproc = dset.copy()
#     dsetproc["leye"] = None
#     dsetproc["reye"] = None
#     dsetproc["overall"] = None

#     for iframe, row in dset.iterrows():
#         fname = row["fname"][2:-2]
#         fnamerequired = respondent + '_' + \
#             prefix + '{:05d}'.format(iframe+1) + fext
#         if fname != fnamerequired:
#             print(" ERROR: frame ", iframe, ": required ",
#                   fnamerequired, ", read ", fname)
#             status = False
#             continue
#         scoring = row["scoring"]
#         if (iframe == 0) and (scoring == '{}'):
#             print(" ERROR: frame ", iframe, " empty")
#             status = False
#             continue
#         if not scoring == "{}":
#             dictval = ast.literal_eval(row["scoring"])
#             logger.info(f"{iframe}, {dictval}")
#             try:
#                 leye = dictval["Left_Eye"]
#                 reye = dictval["Right_Eye"]
#                 overall = dictval["Overall"]
#             except:
#                 logger.error(" ERROR: incomplete gaze direction specification")
#                 status = False
#                 continue
#         # assign scoring
#         dsetproc.iloc[iframe]["leye"] = leye
#         dsetproc.iloc[iframe]["reye"] = reye
#         dsetproc.iloc[iframe]["overall"] = overall

#     dsetproc.index.name = "frame"

#     return dsetproc, status


# def read_tstamps(fname):
#     """Read an annotations file.

#     Codes are the same as for manual scoring
#     via output - 0:"M", 1:"R", 2:"L"
#     tstamps extra codes: 6: transit, 7: end of video

#     Args:
#         fname (string): input filename
#     Returns:
#         tstamps (dataframe): processed annotations

#     """

#     tstamps = pd.read_csv(fname, index_col=0)

#     # recode annotations to match automatic scoring
#     tstamps.replace(2, -1, inplace=True)  # tstamps  L (2) -> dot R (-1)
#     tstamps.replace(6, -2, inplace=True)  # transit  T (6) -> transit -2
#     tstamps.replace(7, -3, inplace=True)  # video end  V (7) -> eof -3

#     return tstamps


# def get_direction(hdirval):
#     """Get gaze direction text from numeric code.

#     Args:
#         hdirval (int): numeric code for the gaze direction
#     Returns:
#         hdir (string): name of the gaze direction

#     """

#     if hdirval == -1:
#         hdir = 'RIGHT'
#     elif hdirval == 0:
#         hdir = 'MIDDLE'
#     elif hdirval == 1:
#         hdir = 'LEFT'
#     elif hdirval == 2:
#         hdir = 'BLINK'
#     elif hdirval == 3:
#         hdir = 'OUTSIDE'
#     elif hdirval == 4:
#         hdir = 'UNKNOWN'

#     return hdir
