import numpy as np
import pandas as pd
import os
import cv2
import dlib
import scipy.signal as signal
import matplotlib.pyplot as plt
from imutils import face_utils
import logging


from GazeScorer import plot_utils
from GazeScorer import settings_utils
from GazeScorer import image_utils


def set_lmarksvars(shape_predictor):
    
    """
    Set indices and colours for face landmarks.

    Args:
        shapre_predictor (str): path to the dlib shape pridictor .dat file
    Returns:
        npnts (number): the number of points in the face landmarks
        flmarksinds (dict): indices for face landmarks
        flmarkscolours (dict): colours for face landmarks
    """

    

    logger = logging.getLogger(__name__)

    if shape_predictor == "shape_predictor_68_face_landmarks.dat":
        npnts = 68

        # indices for face landmarks
        flmarksinds = {}
        flmarksinds["jaw"] = range(0, 17)
        flmarksinds["rbrow"] = range(flmarksinds["jaw"][-1] + 1, 22)
        flmarksinds["lbrow"] = range(flmarksinds["rbrow"][-1] + 1, 27)
        flmarksinds["bnose"] = range(flmarksinds["lbrow"][-1] + 1, 31)
        flmarksinds["fnose"] = range(flmarksinds["bnose"][-1] + 1, 36)
        flmarksinds["reye"] = range(flmarksinds["fnose"][-1] + 1, 42)
        flmarksinds["leye"] = range(flmarksinds["reye"][-1] + 1, 48)
        flmarksinds["omouth"] = range(flmarksinds["leye"][-1] + 1, 60)
        flmarksinds["imouth"] = range(flmarksinds["omouth"][-1] + 1, 68)

        # colours for face landmarks
        flmarkscolours = {}
        flmarkscolours["jaw"] = "w"
        flmarkscolours["rbrow"] = "r"
        flmarkscolours["lbrow"] = "b"
        flmarkscolours["bnose"] = "w"
        flmarkscolours["fnose"] = "w"
        flmarkscolours["reye"] = "r"
        flmarkscolours["leye"] = "b"
        flmarkscolours["omouth"] = "w"
        flmarkscolours["imouth"] = "w"
    else:
        logger.warning(
            f"{shape_predictor} shape predictor is not available to use")
        logger.warning("valid predictors: shape_predictor_68_face_landmarks")
        npnts = None
        flmarksinds = None
        flmarkscolours = None

    return npnts, flmarksinds, flmarkscolours


def get_face(img, fdetector, face_id=None, multiple_id=True):

    logger = logging.getLogger(__name__)

    """
        Gets the bounding box for a face in an image.
        - pre-trained HOG + Linear SVM object detector
          specifically for face detection:
          https://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/

        Args:
            img (cv2 image): input image
            fdetector: the cv2 detector
        Returns:
            fbbox (tuple): bouding box for face (fx, fy, fw, fh)
            frect (array): face rectangle as returned by the detector
    """
    
    

    frectsraw = fdetector(img, 1)
    fcounts = len(frectsraw)
    

    if fcounts == 0:
        # raise ValueError('No faces detected')

        return None, None, face_id

    elif fcounts == 1:

        rect = frectsraw[0]

        farea = None
        fbbox = None
        frect = None

        (fx, fy, fw, fh) = face_utils.rect_to_bb(rect)
        area = fx * fw

        frect = rect
        fbbox = (fx, fy, fw, fh)
        farea = area

        return fbbox, frect, face_id 
    
    if face_id is not None and fcounts > 1:
            # print("Face ID provided. Processing desired face")
            # print(frectsraw)
            # if len(frectsraw) < face_id:
            #     rect = frectsraw
            # else:
            # print(f'len of frectsraw = {len(frectsraw)}')
            rect = frectsraw[face_id]

            farea = None
            fbbox = None
            frect = None

            (fx, fy, fw, fh) = face_utils.rect_to_bb(rect)
            area = fx * fw

            frect = rect
            fbbox = (fx, fy, fw, fh)
            farea = area

            return fbbox, frect, face_id  

    elif multiple_id and fcounts > 1:

        print("More than one face detected. Please select desired face")
        rect, face_id = multiple_face(img, frectsraw)

        farea = None
        fbbox = None
        frect = None

        (fx, fy, fw, fh) = face_utils.rect_to_bb(rect)
        area = fx * fw

        frect = rect
        fbbox = (fx, fy, fw, fh)
        farea = area

        return fbbox, frect, face_id

    else:

        farea = None
        fbbox = None
        frect = None
        face_id = face_id

        # assume that the face of interest in the largest one
        # print('')
        # print(f'len of frectsraw = {len(frectsraw)}')
        for rect in frectsraw:
            (fx, fy, fw, fh) = face_utils.rect_to_bb(rect)
            area = fx * fw
            if frect is None or area > farea:
                frect = rect
                fbbox = (fx, fy, fw, fh)
                farea = area

        return fbbox, frect, face_id
    



def multiple_face(img, frectsraw):
    
    """Function to select desired face if multiple detected
    
        If multiple faces are detected and the multiple_id option is true then this function will raise an image of the first frame with labels for each face detected.
        You can then select which face is desired. The Face id can then be provided in future to bypass this step.
        
        Args
            img (image file): opencv image file for frame
            frectsraw (list): list of face coordiantes identified
            
        Return
            frectsraw (list): list containing the selected face
            face_id (string): corresponding ID for future use  """ 

    # turns on matplot lib interactive mode
    plt.ion()
    
    #start counter for faces
    i_face = 1

    #loops throught each face detected in the frectsraw
    for face in frectsraw: 

        face_id = f"Face_{i_face}"
        y1 = face.bottom()  # detect box bottom y value
        y2 = face.top()  # top y value
        x1 = face.left()  # left x value
        x2 = face.right()  # right x value
        
        # add detected face box in image and adds face id text
        cv2.rectangle(img, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0, 255, 0), 1)
        cv2.putText(
            img, face_id, (x1, y1 -
                           10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        i_face += 1

    # shows image with the faces marked and ID'd
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    face_range = range(1,len(frectsraw))

    #Asks for input for which face. 
    k = input(f"which face would you like to process: {face_range}")
    
    # Grabs selected face data
    face_id = int(k)-1
    frectsraw = frectsraw[face_id]
    

    # print(face_id)
    # print(type(face_id))

    return frectsraw, face_id


def get_flandmarks(img, frect, flpredictor):

    logger = logging.getLogger(__name__)

    """
        Get the landmarks from a face in an image.
        See facial_landmarks_68markup.jpg

        Args:
            img (cv2 image): input image
            frect (array): face rectangle
            flpredictor: trained face landmarks predictor
        Returns:
            lmarks (array): the coordinates for face landmarks

    """
    # get the face landmarks
    lmarks = None
    lmarksraw = flpredictor(img, frect)
    if lmarksraw is not None:
        lmarks = face_utils.shape_to_np(lmarksraw)

    return lmarks


def get_iris(face_eye, elmarks, ieye, eyeplots, do_plot=False):

    logger = logging.getLogger(__name__)

    """
        Get the coordinates for the iris inside an eye region.

        Args:
            face_eye (cv2 image): the image of the eye
            elmarks (array): rectange with eye coordinates
            ieye (number): index for the eye
            eyeplots (dict): settings for eye plots
            do_plot (bool): whether to plot the eye and iris
        Returns:
            shape: iris shape contour
            shaperaw: raw iris shape contour
            face_eye: processed image of the eye
            irep: number of repetitions requited to get the iris
    """

    if do_plot:
        fig, axs = plt.subplots(figsize=(4, 4), nrows=2, ncols=1)
        ax = axs[0]
        ax.imshow(face_eye, cmap="gray", interpolation="bicubic")

    # set the area outside the eye landmarks to black
    stencil = np.zeros(face_eye.shape).astype(face_eye.dtype)
    cv2.fillPoly(stencil, np.array([elmarks], dtype=np.int32), 1)
    face_eye = cv2.bitwise_and(face_eye, stencil)
    if do_plot:
        ax = axs[1]
        ax.imshow(face_eye, cmap="gray", interpolation="bicubic")
        input()

    nreps = 20
    stopreps = False
    psymm = None
    mindiff = 0.01

    # initialise shapes
    shape, shaperaw = None, None
    for irep in range(nreps):
        lshapesraw, hierarchy = cv2.findContours(
            face_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        # print(f'lshaperaw = {len(lshapesraw)}')
        lshape, lshapearea = None, None
        for shape in lshapesraw:
            # loop through the contours to select the most appropriate one
            area = cv2.contourArea(shape)
            if lshape is None or area > lshapearea:
                # no shape stored yet or current shape is larger
                lshape = shape
                lshapearea = area
        # process the selected shape
        stopreps = False
        rect = cv2.boundingRect(lshape)
        x, y, width, height = rect

        areapnts = np.zeros((face_eye.shape[1], face_eye.shape[1]), int)
        noutside = 0
        for i in range(areapnts.shape[0]):
            for j in range(areapnts.shape[1]):
                if len(lshapesraw) == 0:
                    stopreps = True
                elif cv2.pointPolygonTest(lshape, (i, j), True) >= 0 : 
                    areapnts[i, j] = 1
                    noutside += 1
                # elif cv2.pointPolygonTest(lshape, (i, j), True) >= 0:
                #     areapnts[i, j] = 1
                #     noutside += 1

        if stopreps:
            break

        (X, Y) = np.meshgrid(
            range(0, areapnts.shape[1]), range(0, areapnts.shape[0]))
        xcenter = (X * areapnts).sum() / areapnts.sum().astype("float")
        ycenter = (Y * areapnts).sum() / areapnts.sum().astype("float")
        # radius   =  np.min([width, height])/2  # inscribed circle
        radius = np.max([width, height]) / 2  # insribing circle
        symmetry = height / width
        shape = (ycenter, xcenter, radius)
        shaperaw = lshape

        if symmetry > 0.95:
            stopreps = True
        elif (psymm is not None) and (abs(psymm - symmetry) < mindiff):
            stopreps = True
        else:
            psymm = symmetry

        if stopreps:
            break
        if nreps > 1:
            # set the binary image to zero towards the outside edge of the eye
            if ieye == 0:
                for i in range(face_eye.shape[0]):
                    for j in range(face_eye.shape[1]):
                        if j > shape[0] + shape[2]:
                            face_eye[i, j] = 0
            else:
                for i in range(face_eye.shape[0]):
                    for j in range(face_eye.shape[1]):
                        if j < shape[0] - shape[2]:
                            face_eye[i, j] = 0

    return shape, shaperaw, face_eye, irep


def process_frames(
    framelist,
    settings=None,
    ind4calibonset=0,
    face_id=None,
    multiple_id=True,
    PID=None,
    plot_face_lmarks=None,
    plot_face_lmarks_all=None,
    plot_iris_lmarks=None,
    plot_iris_lmarks_all=None,
):
    """Function to process the frames to locate the coordinates for the face and iris
    
        Uses the dlib frontal face detecor to ID the faces in the frames and mark their x-y coords.
        Using the identified eye coordiantes an estimate of the Iris is made and plotted. 
        The centre of the iris is then identifed for etimating gaze location.
        
    Args:
        framelist (list): list of opencv images for each frame
        settings (dict): dictionary of settings from setting functions
        ind4calibonset (int): index for which frame an established first look is (typically rhe first frame)
        face_id (string): identifier for the desired face to process
        multiple_id (bool): boolean to identify if multiple id procedure to be followed. If false, largest face will be used
        PID (str): particiapnt identifier
        plot_face_lmarks (bool): creates an image of the first frame with the face plotted
        plot_face_lmarks_all (bool):  creates an image of all frames with the face plotted
        plot_iris_lmarks (bool): creates cropped image of the plotted iris for the first frame
        plot_iris_lmarks_all (bool):  creates cropped image of the plotted iris for all frames

    Return:
        etdset (dataframe): dataframe of the coordinates for the face and iris. Each row corresponds to a frame

    """


    logger = logging.getLogger(__name__)

    logger.info("PROCESSING INDIVIDUAL VIDEO FRAMES")

    if settings is None:
        settings = settings_utils.assign_settings()

    proc = settings["proc"]
    gaze = settings["gaze"]
    plots = settings["plots"]

    # initialize dlib's face detector (HOG-based)
    face_detector = dlib.get_frontal_face_detector()

    # sets face detector error variable
    i_face_error = 0
    error_limit = (proc["percent_error_limit"] / 100) * len(framelist)
    print(f"error limit = {error_limit}")

    # deal with multiple faces
    # if face_id is None:
    #     face_id = 0

    # create the facial landmark predictor
    flmarks_predictor = dlib.shape_predictor(proc["shape_predictor"])
    npnts, flmarksinds, flmarkscolours = image_utils.set_lmarksvars(
        proc["shape_predictor"]
    )
    if npnts is None:
        return

    # create columns for dataframe eye features output
    etcolumns = [
        "fx",
        "fy",
        "fw",
        "fh",
        "elx",
        "ely",
        "elw",
        "elh",
        "ilx",
        "ily",
        "ilr",
        "ilireps",
        "erx",
        "ery",
        "erw",
        "erh",
        "irx",
        "iry",
        "irr",
        "irireps",
    ]
    for lmark in flmarksinds:
        for ind in flmarksinds[lmark]:
            # AF Inverted X and Y to fix error in plotting visual on face image
            etcolumns.extend(
                [
                    lmark + str(ind - flmarksinds[lmark][0]) + "x",
                    lmark + str(ind - flmarksinds[lmark][0]) + "y",
                ]
            )

    do_verbose = False

    etdset = pd.DataFrame(columns=etcolumns)
    fail_face_count = 0

    for iframe, img in enumerate(framelist):
        logger.info(f"frame: {img}")

        try:

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # if iframe > 0 and multiple_id:
            #     multiple_id = False


        
            # get the bounding box for the face
            fbbox, frect, face_id = get_face(
                img, face_detector, face_id, multiple_id)
            # print(fbbox)
            # print(frect)
            

            # initialise variables for outputs
            l_et = np.empty((1, len(etdset.columns)))
            l_et[:] = np.NaN
            flmarks = None
            ebboxes = {}
            ecoords = {}
            iris = {}
            irisraw = {}
            eye_img = {}
            eye_img_iris = {}
            eye_img_proc = {}
            annotations = {}

            while fbbox is not None and all(iris.values()):
                (fx, fy, fw, fh) = fbbox
                l_et[0, : len(fbbox)] = fbbox  # face coordinates
                etind = len(fbbox)  # index for next data
                flmarks = get_flandmarks(img, frect, flmarks_predictor)
                if flmarks is not None:
                    for ieye, eye in enumerate(gaze["elist"]):
                        ecoords[eye] = flmarks[flmarksinds[eye[0] + "eye"]].copy()
                        ex = ecoords[eye][0, 0]
                        ey = np.min([ecoords[eye][1, 1], ecoords[eye][2, 1]])
                        ew = ecoords[eye][3, 0] - ex
                        eh = np.max([ecoords[eye][4, 1], ecoords[eye][5, 1]]) - ey
                        # add extra vertical buffer to the eye
                        ehbuf = int(np.round(proc["ehbufratio"] * eh))
                        ey -= ehbuf
                        eh += 2 * ehbuf
                        # save the eye coordinates
                        ebboxes[eye] = (ex - fx, ey - fy, ew, eh)
                        # get iris
                        face_img = img[fy: fy + fh, fx: fx + fw]
                        eye_img[eye] = img[ey: ey + eh, ex: ex + ew]
                        elmarks = ecoords[eye].copy()
                        elmarks = np.vstack((elmarks, elmarks[0, :]))
                        elmarks[:, 0] -= ex
                        elmarks[:, 1] -= ey

                        # process the eye image
                        eye_img_iris[eye] = cv2.medianBlur(eye_img[eye], 7)
                        eye_img_iris[eye] = ~eye_img_iris[eye]

                        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(
                            eye_img_iris[eye])
                        thresh = np.round(minVal + 0.50 * (maxVal - minVal))
                        title = [
                            eye,
                            ":",
                            str(int(np.rint(minVal))),
                            ", ",
                            str(int(np.rint(maxVal))),
                            ", ",
                            str(int(thresh)),
                            ", ",
                            str(minLoc),
                            ", ",
                            str(maxLoc),
                        ]
                        ret, eye_img_iris[eye] = cv2.threshold(
                            eye_img_iris[eye], thresh, 255, cv2.THRESH_BINARY
                        )

                        ## TO DO - ADD IN ANNOTATIONS ##

                        # if plots["do_annotations"] and (img == ind4calibonset):
                        #     annotations[eye] = (
                        #         irisa[eye][0] - ex, irisa[eye][1] - ey, irisa[eye][2])
                        # else:
                        #     annotations[eye] = None
                        
                        iris[eye], irisraw[eye], eye_img_proc[eye], irep = get_iris(
                            eye_img_iris[eye],
                            elmarks,
                            gaze["elist"].index(eye),
                            plots,
                            do_plot=False,
                        )
                        # save data to dataframe
                        l_et[0, etind: etind + 4] = ebboxes[eye]
                        etind += 4
                        l_et[0, etind: etind + 3] = iris[eye]
                        etind += 3
                        l_et[0, etind] = irep
                        etind += 1

                    # print(iris)
                    # print(type(iris))
                    # TO DO - Add in plots ###

                    # if plots["do_annotations"] and (iframe == ind4calibonset):
                    #     plotfname = annotatefolder + resp + \
                    #         "_annotations_{:03d}.png".format(iframe)
                    #     plot_utils.plot_annotations(
                    #         eye_img, eye_img_iris, gaze["elist"], annotations, plots, plotfname)

                    # if plot_iris_lmarks is not None plot_iris_lmarks_all is not None:

                    if plot_iris_lmarks and iframe == 0:
                        if PID is not None:
                            plotfname = (
                                f"{plot_iris_lmarks}/{PID}_iris_lmarks{(iframe+1):03d}.png"
                            )
                        else:
                            plotfname = (
                                f"{plot_iris_lmarks}/iris_lmarks{(iframe+1):03d}.png"
                            )

                        plot_utils.plot_iris(
                            eye_img,
                            eye_img_proc,
                            annotations,
                            elmarks,
                            iris,
                            irisraw,
                            gaze["elist"],
                            plots,
                            plotfname,
                        )
                    elif plot_iris_lmarks_all is not None:

                        if PID is not None:
                            plotfname = (
                                f"{plot_iris_lmarks}/{PID}_iris_lmarks{iframe:03d}.png"
                            )
                        else:
                            plotfname = f"{plot_iris_lmarks}/iris_lmarks{iframe:03d}.png"

                        plot_utils.plot_iris(
                            eye_img,
                            eye_img_proc,
                            annotations,
                            elmarks,
                            iris,
                            irisraw,
                            gaze["elist"],
                            plots,
                            plotfname,
                        )

                    flmarks[:, 0] -= fx
                    flmarks[:, 1] -= fy
                    l_et[0, etind:] = np.reshape(flmarks, (1, flmarks.size))

                    break

            # Checks if failed to detext face. returns exception if error
            else:
                i_face_error += 1

                print(f"error frame = {iframe}")
                print(f"error count = {i_face_error}")

                if iframe < (proc["nadjustmentpnts"]) and i_face_error > (
                    round(proc["nadjustmentpnts"] / 2)
                ):
                    raise RuntimeError(
                        f"face not detected in {round(proc['nadjustmentpnts'] / 2)} of {proc['nadjustmentpnts']} intial calibration frames. Processing has been terminated"
                    )
                if i_face_error > error_limit:
                    raise RuntimeError(
                        f"face not detected for {proc['percent_error_limit']} percent of frames. Processing has been terminated at frame {iframe+1}/{len(framelist)+1}"
                    )

                l_et[:] = -100


                # break

            # append frame data
            etdset = pd.concat([etdset, pd.DataFrame(l_et, columns=etdset.columns)],
                ignore_index=True,
            )

            #### TO DO - add in optional plot functions###
            # print(all(iris.values()))
            if fbbox is not None and all(iris.values()):
                if plot_face_lmarks and iframe == 0:

                    if PID is not None:
                        plotfname = f"{plot_face_lmarks}/{PID}_face_lmarks{(iframe+1):03d}.png"
                    else:
                        plotfname = f"{plot_face_lmarks}/face_lmarks{(iframe+1):03d}.png"

                    plot_utils.plot_facelmarks(
                        img,
                        annotations,
                        fbbox,
                        flmarks,
                        flmarksinds,
                        flmarkscolours,
                        gaze["elist"],
                        ebboxes,
                        plots,
                        iris,
                        plotfname,
                        plot_whole=plots["do_whole"],
                    )

                elif plot_face_lmarks_all is not None:

                    if PID is not None:
                        plotfname = (
                            f"{plot_face_lmarks_all}/{PID}_face_lmarks{(iframe+1):03d}.png"
                        )
                    else:
                        plotfname = f"{plot_face_lmarks_all}/face_lmarks{(iframe+1):03d}.png"

                    plot_utils.plot_facelmarks(
                        img,
                        annotations,
                        fbbox,
                        flmarks,
                        flmarksinds,
                        flmarkscolours,
                        gaze["elist"],
                        ebboxes,
                        plots,
                        iris,
                        plotfname,
                        plot_whole=plots["do_whole"],
                    )

        except:
            
            raise

    logger.info("Done")
    etdset.index.name = "frame"

    
    return etdset


def process_lmarks(
    etdset,
    settings=None,
    ind4calibonset=0,
    PID=None,
    do_phcoords=None,
    do_hdir=None,
    do_aratio=None,
):

    """Processes the landmarks to give an estimated gaze orientation
    
        This function takes in the etdset of the face and iris landmarks and estimates wherer the iris centre is 
        and establishes an estimate of the gaze orientation
        
    Args:
        etdset (dataframe): dataframe of face and iris landmarks
        settings (dict): dictionary of settings
        ind4calibonset (int): index of the initial gaze location
        PID (str): participant identifier
        do_phcoords (bool): boolean to identify if phcoords will be plotted
        do_hdir (bool): boolean to identify if hdir will be plotted
        do_aratio (bool): boolean to identify if aratios should be plotted
        
    Returns:
         etdset (dataframe): Updated data frame with gaze orientation data
         aratios (dtaframe): aratios of aspect ratio for each eye. used for distinguishing blinks
        """

    logger = logging.getLogger(__name__)

    logger.info("ANNOTATING FRAMES WITH PLOTTED DIRECTIONS")

    if settings is None:
        settings = settings_utils.assign_settings()

    proc = settings["proc"]
    gaze = settings["gaze"]
    plots = settings["plots"]

    nframes = etdset.shape[0]
    # print(f'nframes:{nframes}')
    i_blink_error = 0
    error_limit = (proc["percent_error_limit"] / 100) * (nframes)

    # calculate various eye measures
    aratios = {}
    for eye in gaze["elist"]:
        # aspect ratio
        etdset["e" + eye[0] + "aratio"] = (
            etdset["e" + eye[0] + "h"] / etdset["e" + eye[0] + "w"]
        )
        aratios["e" + eye[0] +
                "mean"] = np.mean(etdset["e" + eye[0] + "aratio"].values)
        aratios["e" + eye[0] +
                "std"] = np.std(etdset["e" + eye[0] + "aratio"].values)
        # calculate x and y middle values
        etdset["e" + eye[0] + "xmidraw"] = etdset["e" + eye[0] + "w"] / 2
        etdset["e" + eye[0] + "ymidraw"] = etdset["e" + eye[0] + "h"] / 2
    etdset["aratioavg"] = etdset.loc[:, ["elaratio", "eraratio"]].mean(axis=1)
    aratios["avgmean"] = np.mean(etdset["aratioavg"].values)
    aratios["avgstd"] = np.std(etdset["aratioavg"].values)

    if proc["do_rounding"]:
        for column in proc["columns2round"]:
            etdset[column] = etdset[column].apply(np.round)

    if proc["do_filtering"]:
        for column in proc["columns2filter"]:
            etdset[column] = signal.medfilt(
                etdset[column].values, proc["nflt"])

    eyemidxoffsets = {}
    eyemidxstds = {}

    for eye in gaze["elist"]:
        frames = np.arange(ind4calibonset, ind4calibonset +
                           proc["nadjustmentpnts"], 1)
        irisx = etdset.loc[:, "i" + eye[0] + "x"]
        emidx = etdset.loc[:, "e" + eye[0] + "xmidraw"]
        
        if len(frames) > len(irisx):
            raise RuntimeError('Not enought frames in video for initial calibration')
        
        ixcoords = irisx.loc[frames]
        emidxcoords = emidx.loc[frames]
        if proc["do_adjustment"]:
            # horizontal calibration
            # calculate the calilbration offset and standard deviation
            eyemidxoffsets[eye] = ixcoords.mean() - emidxcoords.mean()
            eyemidxstds[eye] = ixcoords.std()
        else:
            eyemidxoffsets[eye] = 0
            eyemidxstds[eye] = ixcoords.std()

        # calculate the calibrated eye middle x coordinates
        etdset["e" + eye[0] + "xmid"] = (
            etdset["e" + eye[0] + "xmidraw"] + eyemidxoffsets[eye]
        )
        etdset["i" + eye[0] + "dx"] = (
            etdset.loc[:, "i" + eye[0] + "x"] - etdset["e" + eye[0] + "xmid"]
        )
        if proc["emidbuf"] == "std":
            etdset["e" + eye[0] + "xmidbuf"] = proc["emidstdratio"] * \
                eyemidxstds[eye]
        elif proc["emidbuf"] == "ratio":
            etdset["e" + eye[0] + "xmidbuf"] = (
                proc["emidwdtratio"] * etdset["e" + eye[0] + "w"]
            )

        etdset["e" + eye[0] + "xmidmin"] = (
            etdset["e" + eye[0] + "xmid"] - etdset["e" + eye[0] + "xmidbuf"]
        )
        etdset["e" + eye[0] + "xmidmax"] = (
            etdset["e" + eye[0] + "xmid"] + etdset["e" + eye[0] + "xmidbuf"]
        )

    # add empty columns for looks and other calculated variables
    # <eye> - l: left, r: right
    # e<eye>hdirval - numerical code for the horizontal gaze direction for <eye>,  -1: left, 0: middle, 1: right
    # e<eye>hdir - horizontal look direction: LEFT/MIDDLE/RIGHT/BLINK
    # e<eye>hdirx - horizontal excursion for <eye> in fractional pixels, negative value: left, positive value: right
    # hdirval - numerical code for overall horizontal gaze direction,  -1: left, 0: middle, 1: right
    # hdir - overall horizontal gaze direction: LEFT/MIDDLE/RIGHT
    # hdirx - overall horizontal excursion in fractional pixels, negative value: left, positive value: right
    # hdirconf - confidence value for overall horizontal gaze direction
    for eye in gaze["elist"]:
        etdset["e" + eye[0] + "hdirval"] = ""
    etdset["hdirval"] = ""
    etdset["hdirx"] = ""
    etdset["hdirdx"] = ""
    etdset["hdirconf"] = ""
    etdset["hfxt"] = 0

    # first pass through the dataset to decide looks based on information from current frame
    for iframe, etrow in etdset.iterrows():
        (fx, fy, fw, fh) = etrow["fx"], etrow["fy"], etrow["fw"], etrow["fh"]
        if ~np.isnan(fx):
            # decide looks for each eye
            emids = {}
            for eye in gaze["elist"]:
                (ex, ey, ew, eh) = (
                    etrow["e" + eye[0] + "x"],
                    etrow["e" + eye[0] + "y"],
                    etrow["e" + eye[0] + "w"],
                    etrow["e" + eye[0] + "h"],
                )
                val2set = "e" + eye[0] + "hdirval"
                if proc["emidbuf"] == "none":
                    # no buffer around the middle
                    if etrow["i" + eye[0] + "x"] > etrow["e" + eye[0] + "xmid"]:
                        etdset.loc[iframe, val2set] = 1
                    else:
                        etdset.loc[iframe, val2set] = -1
                else:
                    # buffer around the middle
                    etdset.loc[iframe, val2set] = 0  # default middle
                    emids[eye] = (
                        etrow["e" + eye[0] + "xmidmin"],
                        etrow["e" + eye[0] + "xmidmax"],
                    )
                    if etrow["i" + eye[0] + "x"] > emids[eye][1]:
                        etdset.loc[iframe, val2set] = 1
                    elif etrow["i" + eye[0] + "x"] < emids[eye][0]:
                        etdset.loc[iframe, val2set] = -1

                # mark blinks for each eye
                if proc["do_blinks"]:
                    if (
                        etrow["e" + eye[0] + "aratio"]
                        < aratios["e" + eye[0] + "mean"] - aratios["e" + eye[0] + "std"]
                    ):
                        etdset.loc[iframe, "e" + eye[0] + "hdirval"] = gaze["hdirvals"][gaze["hdirlabels"].index("BLINK")]

    # filter the horizontal gaze directions
    if proc["do_filtering"]:
        for eye in gaze["elist"]:
            etdset.loc[:, "e" + eye[0] + "hdirval"] = signal.medfilt(
                etdset.loc[:, "e" + eye[0] + "hdirval"], proc["nflt"]
            )

    # fixation variables
    phidx = None
    nhdirx = 0
    for iframe, etrow in etdset.iterrows():
        # decide overall look direction
        hdirval = gaze["hdirs"]["UNKNOWN"]
        # utility variable
        elhdirval = etrow["elhdirval"]
        erhdirval = etrow["erhdirval"]
        elhx = etrow["ilx"]
        erhx = etrow["irx"]
        if elhdirval == "" or erhdirval == "":
            etdset.loc[iframe, "hdirval"] = hdirval
            etdset.loc[iframe, "hdirx"] = hdirx
            etdset.loc[iframe, "hdirdx"] = hdirdx
            continue
        # horizontal gaze direction values
        elhdir = gaze["hdirlabels"][gaze["hdirvals"].index(elhdirval)]
        erhdir = gaze["hdirlabels"][gaze["hdirvals"].index(erhdirval)]
    
        if elhdir in gaze["hdirlabels"] and erhdir in gaze["hdirlabels"]:
            if elhdir == erhdir:
                # horizontal direction for the two eyes are the same
                hdirval = elhdirval
                hdir = gaze["hdirlabels"][gaze["hdirvals"].index(hdirval)]
                # for x use right eye for right look (temporal gaze) and left eye for left look (nasal gaze)
                if hdir == "MIDDLE":
                    hdirx = 0
                    hdirdx = 0
                elif hdir == "LEFT" or hdir == "RIGHT":
                    eye2use = hdir
                    # hdirx = etrow["e"+eye2use.lower()[0]+"hdirx"]
                    hdirx = etrow["i" + eye2use.lower()[0] + "x"]
                    hdirdx = etrow["i" + eye2use.lower()[0] + "dx"]
                elif elhdir == "BLINK" or erhdir == "BLINK":
                    
                    i_blink_error += 1

                    if iframe < (proc["nadjustmentpnts"]) and i_blink_error > (
                                                    round(proc["nadjustmentpnts"] / 2)
                                                ):
                        raise RuntimeError(
                            f"Blinks detected in half of the intial {proc['nadjustmentpnts']} calibration frames. Processing has been terminated"
                        )
                    

                    hdirval = gaze["hdirvals"][gaze["hdirlabels"].index("BLINK")]
                    hdirx = gaze["hdirvals"][gaze["hdirlabels"].index("BLINK")]
                    hdirdx = gaze["hdirvals"][gaze["hdirlabels"].index("BLINK")]
            elif (elhdir == "RIGHT" and erhdir == "LEFT") or (
                elhdir == "LEFT" and erhdir == "RIGHT"
            ):
                # the eyes are "crossing", means overall look is middle
                hdirval = 0
                hdirx = 0
                hdirdx = 0
            elif elhdir == "MIDDLE" and erhdir == "RIGHT":
                # left eye middle, use right eye for overall direction
                hdirval = erhdirval
                hdirx = etrow["irx"]
                hdirdx = etrow["irdx"]
            elif erhdir == "MIDDLE" and elhdir == "LEFT":
                # right eye is middle, use left eye for overall direction
                hdirval = elhdirval
                hdirx = etrow["ilx"]
                hdirdx = etrow["ildx"]
            elif (elhdir == "MIDDLE" and erhdir == "LEFT") or (
                erhdir == "MIDDLE" and elhdir == "RIGHT"
            ):
                hdirval = 0
                hdirx = 0
                hdirdx = 0

        etdset.loc[iframe, "hdirval"] = hdirval
        etdset.loc[iframe, "hdirx"] = hdirx
        etdset.loc[iframe, "hdirdx"] = hdirdx

        # judge fixation based on the overall direction
        if phidx == None:
            phidx = hdirdx
            nhdirx = 1
            etdset.loc[iframe, "hfxt"] = 0
        else:
            if phidx == hdirdx:
                nhdirx += 1
                if nhdirx > proc["n4fxt"]:
                    etdset.loc[iframe - nhdirx: iframe + 1, "hfxt"] = 1
            else:
                phidx = hdirdx
                nhdirx = 0

    
    aratios = pd.DataFrame(aratios, index=[0])

    return etdset, aratios
