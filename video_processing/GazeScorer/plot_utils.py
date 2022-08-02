from logging import getLogger
import numpy as np
import scipy.signal as signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import logging
import plotnine as p9
from plotnine import *
import statistics


import warnings

import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype



def axes_setvisibility(ax, visibility):
    """
    Set visibillity for the matplotlib spines

    Args:
        visibility - [left, top, right, bottom]
    Returns:
        none
    """

    ax.spines["left"].set_visible(visibility[0])
    ax.spines["top"].set_visible(visibility[1])
    ax.spines["right"].set_visible(visibility[2])
    ax.spines["bottom"].set_visible(visibility[3])

    if not visibility[0]:
        ax.set_yticks([])
        ax.set_yticklabels([])

    if not visibility[3]:
        ax.set_xticks([])
        ax.set_xticklabels([])


def plot_iris(
    eye_img, eye_img_proc, irisa, elmarks, iris, irisraw, eyeslist, plots, fname
):
    """
    Plot the eye with iris information.

    Args:

    Returns:
        none
    """

    fig, axs = plt.subplots(figsize=(8, 4), nrows=2, ncols=2)

    for ieye, eye in enumerate(eyeslist):
        # plot eyes for pupil
        ax = axs[0, (ieye + 1) % 2]
        ax.axis("off")
        ax.imshow(eye_img[eye], cmap="gray", interpolation="bicubic")
        ax.plot(elmarks[:, 0], elmarks[:, 1], color="g", linewidth=4)
        if irisraw is not None:
            pnts2plot = np.array([[pnt[0, 0], pnt[0, 1]]
                                 for pnt in irisraw[eye]])
            ax.plot(pnts2plot[:, 0], pnts2plot[:, 1],
                    linewidth=plots["elinewdt"])

            ax.add_patch(
                mpl.patches.Circle(
                    (iris[eye][0], iris[eye][1]),
                    iris[eye][2],
                    linewidth=plots["elinewdt"],
                    edgecolor=plots["ecolours"][ieye],
                    facecolor="none",
                )
            )
        if bool(irisa):
            if (irisa["left"] is not None) and (irisa["right"] is not None):
                ax.add_patch(
                    mpl.patches.Circle(
                        (irisa[eye][0], irisa[eye][1]),
                        irisa[eye][2],
                        linewidth=plots["elinewdt"],
                        linestyle="--",
                        edgecolor=plots["ecolours"][ieye],
                        facecolor="none",
                    )
                )

        # plot processed eyes for pupil
        ax = axs[1, (ieye + 1) % 2]
        ax.axis("off")
        ax.imshow(eye_img_proc[eye], cmap="gray", interpolation="bicubic")
        if irisraw is not None:
            pnts2plot = np.array([[pnt[0, 0], pnt[0, 1]]
                                 for pnt in irisraw[eye]])
            ax.plot(pnts2plot[:, 0], pnts2plot[:, 1],
                    linewidth=plots["elinewdt"])

            ax.add_patch(
                mpl.patches.Circle(
                    (iris[eye][0], iris[eye][1]),
                    iris[eye][2],
                    linewidth=plots["elinewdt"],
                    edgecolor=plots["ecolours"][ieye],
                    facecolor="none",
                )
            )
        if bool(irisa):
            if (irisa["left"] is not None) and (irisa["right"] is not None):
                ax.add_patch(
                    mpl.patches.Circle(
                        (irisa[eye][0], irisa[eye][1]),
                        irisa[eye][2],
                        linewidth=plots["elinewdt"],
                        linestyle="--",
                        edgecolor=plots["ecolours"][ieye],
                        facecolor="none",
                    )
                )
    plt.tight_layout()
    plt.savefig(fname, dpi=plots["dpi"])
    plt.close()


def plot_facelmarks(
    img,
    irisa,
    fbbox,
    flmarks,
    flmarksinds,
    flmarkscolours,
    eyeslist,
    ebboxes,
    plots,
    iris,
    fname,
    plot_whole=True,
):
    """
    Plot the face with landmarks.

    Args:

    Returns:
        none
    """

    fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
    ax.axis("off")
    img2plot = img
    
    ebboxes2plot = {}
    iris2plot = {}
    irisa2plot = {}
    if plot_whole:
        if fbbox is not None:
            flmarks2plot = flmarks.copy()
            (fx, fy, fw, fh) = fbbox
            flmarks2plot[:, 0] += fx
            flmarks2plot[:, 1] += fy
    else:
        if fbbox is not None:
            (fx, fy, fw, fh) = fbbox
            img2plot = img[fy: fy + fh, fx: fx + fw]
            flmarks2plot = flmarks.copy()
    if fbbox is not None:
        for ieye, eye in enumerate(eyeslist):
            if plot_whole:
                ebboxes2plot[eye] = (
                    ebboxes[eye][0] + fx,
                    ebboxes[eye][1] + fy,
                    ebboxes[eye][2],
                    ebboxes[eye][3],
                )
                iris2plot[eye] = (
                    iris[eye][0] + ebboxes2plot[eye][0] + fx,
                    iris[eye][1] + ebboxes2plot[eye][1] + fy,
                    iris[eye][2],
                )
            else:
                ebboxes2plot[eye] = (
                    ebboxes[eye][0],
                    ebboxes[eye][1],
                    ebboxes[eye][2],
                    ebboxes[eye][3],
                )
                iris2plot[eye] = (
                    iris[eye][0] + ebboxes2plot[eye][0],
                    iris[eye][1] + ebboxes2plot[eye][1],
                    iris[eye][2],
                )

                if bool(irisa):
                    if irisa[eye] is not None:
                        irisa2plot[eye] = (
                            irisa[eye][0] - fx,
                            irisa[eye][1] - fy,
                            irisa[eye][2],
                        )
    img2plot = cv2.cvtColor(img2plot, cv2.COLOR_GRAY2RGB)
    ax.imshow(img2plot)
    if fbbox is not None and flmarks is not None:
        for lmark in flmarksinds:
            pnts2plot = flmarks2plot[flmarksinds[lmark], :]
            if lmark in ["leye", "reye", "omouth", "imouth"]:
                pnts2plot = np.vstack((pnts2plot, pnts2plot[0, :]))
            ax.plot(
                pnts2plot[:, 0],
                pnts2plot[:, 1],
                color=flmarkscolours[lmark],
                marker="*",
            )
    if ebboxes2plot:
        for ieye, eye in enumerate(eyeslist):
            (ex, ey, ew, eh) = ebboxes2plot[eye]
            eyecolour = plots["ecolours"][ieye]
            ax.add_patch(
                mpl.patches.Rectangle(
                    (ex, ey),
                    ew,
                    eh,
                    linewidth=plots["flinewdt"],
                    edgecolor=eyecolour,
                    facecolor="none",
                )
            )
            ax.add_patch(
                mpl.patches.Circle(
                    (iris2plot[eye][0], iris2plot[eye][1]),
                    iris2plot[eye][2],
                    linewidth=2 * plots["flinewdt"],
                    edgecolor=plots["ecolours"][ieye],
                    facecolor="none",
                )
            )
            if bool(irisa):
                if irisa[eye] is not None:
                    ax.add_patch(
                        mpl.patches.Circle(
                            (irisa2plot[eye][0], irisa2plot[eye][1]),
                            irisa2plot[eye][2],
                            linestyle="--",
                            linewidth=plots["flinewdt"],
                            edgecolor=flmarkscolours["jaw"],
                            facecolor="none",
                        )
                    )
        # save figure
    plt.tight_layout()
    plt.savefig(fname, dpi=plots["dpi"])
    plt.close()


def plot_annotations(eye_img, eye_img_iris, eyeslist, annotations, plots, fname):
    fig, axs = plt.subplots(figsize=(8, 4), nrows=2, ncols=2)

    for ieye, eye in enumerate(eyeslist):
        ax = axs[0, (ieye + 1) % 2]
        ax.imshow(eye_img[eye], cmap="gray", interpolation="bicubic")
        ax.add_patch(
            mpl.patches.Circle(
                (annotations[eye][0], annotations[eye][1]),
                annotations[eye][2],
                linewidth=plots["elinewdt"],
                linestyle="--",
                edgecolor=plots["ecolours"][ieye],
                facecolor="none",
            )
        )
        # ax.set_title("".join(title))

        ax = axs[1, (ieye + 1) % 2]
        ax.imshow(eye_img_iris[eye], cmap="gray", interpolation="bicubic")
        ax.add_patch(
            mpl.patches.Circle(
                (annotations[eye][0], annotations[eye][1]),
                annotations[eye][2],
                linewidth=plots["elinewdt"],
                linestyle="--",
                edgecolor=plots["ecolours"][ieye],
                facecolor="none",
            )
        )
    # save figure
    plt.tight_layout()
    plt.savefig(fname, dpi=plots["dpi"])
    plt.close()


def plot_direction(hehdirval, colour, shapes, ax):
    """
    Plot arrows for gaze direction on specified axes.

    Args:

    Returns:
        none
    """

    if hehdirval == -1:
        # RIGHT
        ax.arrow(
            shapes["arrowrightx"],
            shapes["arrowy"],
            -shapes["arrowlength"],
            0,
            color=colour,
            width=2,
        )
    elif hehdirval == 1:
        # LEFT
        ax.arrow(
            shapes["arrowleftx"],
            shapes["arrowy"],
            shapes["arrowlength"],
            0,
            color=colour,
            width=2,
        )
    elif hehdirval == 0:
        # MIDDLE
        ax.add_patch(
            mpl.patches.Circle(
                (shapes["shapemidx"], shapes["arrowy"]),
                4,
                linewidth=4,
                edgecolor=colour,
                facecolor=colour,
            )
        )
    elif hehdirval == 2:
        # BLINK
        ax.add_patch(
            mpl.patches.Ellipse(
                (shapes["shapemidx"], shapes["arrowy"]),
                shapes["ellipsewidth"],
                shapes["ellipseheight"],
                fill=True,
                linewidth=4,
                edgecolor=colour,
                facecolor=colour,
            )
        )
    else:
        # UNKNOWN/OUTSIDE
        ax.add_patch(
            mpl.patches.Ellipse(
                (shapes["shapemidx"], shapes["arrowy"]),
                shapes["ellipsewidth"],
                shapes["ellipseheight"],
                fill=True,
                linewidth=4,
                edgecolor=colour,
                facecolor=colour,
            )
        )


def plot_phcoords(etdset, resp, elist, plots, plotfname):
    """
    Plot horizontal pupil coordinates.

    Args:

    Returns:
        none
    """

    # plot horizontal pupil coordinates
    fig, axs = plt.subplots(figsize=(20, 10), nrows=2, ncols=1)
    finds = list(etdset.index)

    for ieye, eye in enumerate(elist):
        ax = axs[ieye]
        irisx = etdset.loc[:, "i" + eye[0] + "x"]
        emidx = etdset.loc[:, "e" + eye[0] + "xmid"]
        emidxmin = etdset.loc[:, "e" + eye[0] + "xmidmin"]
        emidxmax = etdset.loc[:, "e" + eye[0] + "xmidmax"]
        emidxraw = etdset.loc[:, "e" + eye[0] + "xmidraw"]
        ax.fill_between(finds, emidxmin, emidxmax, color="lightgrey")
        ax.scatter(finds, irisx, c=plots["ecolours"][ieye], s=8, marker=8)
        ax.plot(finds, emidxraw, c=plots["ecolours"][ieye], linestyle="--")
        ax.plot(finds, emidx, c=plots["ecolours"][ieye])
        ax.set_ylim(0, etdset["e" + eye[0] + "w"].max())
        axes_setvisibility(ax, [True, False, False, True])
        ax.legend(
            [
                "Raw eye center x",
                "Adjusted eye center x",
                "Eye center scatter",
                "Iris center x",
            ],
            loc="upper center",
            ncol=4,
        )
        # add horizontal lines to mark the onset/offset for calibration points
        ax.set_title(resp + ": " + eye.upper() + " eye")
    # save the plot to file
    plt.tight_layout()
    # plt.savefig(plotfname, dpi=plots["dpi"])
    plt.savefig(plotfname, dpi=plots["dpi"], ax=ax)
    plt.close()


def plot_hdir(etdset, resp, elist, plots, plotfname):
   
    """
    Plot horizontal gaze direction.

    Args:

    Returns:
        none
    """

    # plot the overall direction
    fig, ax = plt.subplots(figsize=(20, 4), nrows=1, ncols=1)
    finds = etdset.index
    # 
    #ax.plot(etdset.index, etdset["hfxt"])
    ax.scatter(etdset.index, etdset["agazeraw"])
    ax.scatter(etdset.index, etdset["agazeraw_dx"])
    axes_setvisibility(ax, [True, False, False, True])
    ax.set_title(resp, fontsize=12)

    # save the plot to file
    plt.tight_layout()
    plt.savefig(plotfname, dpi=plots["dpi"], ax=ax)
    plt.close()


def plot_aratio(etdset, aratios, resp, elist, plots, plotfname):
    """
    Plot eye aspect ratio.

    Args:

    Returns:
        none
    """

    fig, ax = plt.subplots(figsize=(20, 10), nrows=1, ncols=1)
    finds = etdset.index
    offsets = [-2 * aratios["elmean"], 0]
    for ieye, eye in enumerate(elist):
        ax.plot(
            finds,
            offsets[ieye] + etdset["e" + eye[0] + "aratio"],
            plots["ecolours"][ieye] + plots["emarkers"][ieye] + "-",
        )
        ax.fill_between(
            finds,
            offsets[ieye]
            + (aratios["e" + eye[0] + "mean"] - aratios["e" + eye[0] + "std"]),
            offsets[ieye]
            + (aratios["e" + eye[0] + "mean"] + aratios["e" + eye[0] + "std"]),
            color="lightgrey",
            alpha=0.4,
        )
    offset = -aratios["avgmean"]
    ax.plot(finds, offset + etdset["aratioavg"], "g*-")
    ax.fill_between(
        finds,
        offset + (aratios["avgmean"] - aratios["avgstd"]),
        offset + (aratios["avgmean"] + aratios["avgstd"]),
        color="lightgrey",
        alpha=0.4,
    )

    axes_setvisibility(ax, [False, False, False, True])
    ax.set_yticks([])
    ax.set_xlabel("Frame")
    ax.set_title(resp, fontsize=12)
    ax.legend(
        ["Left eye", "Right eye", "Average", "Mean+/-std"],
        loc="upper left",
        title="Aspect ratios",
        ncol=4,
        fontsize=12,
    )
    for ieye, eye in enumerate(elist):
        ax.plot(
            finds,
            offsets[ieye] + signal.medfilt(etdset["e" + eye[0] + "aratio"], 9),
            plots["ecolours"][ieye] + "-",
        )
    ax.plot(finds, offset + signal.medfilt(etdset["aratioavg"], 9), "g-")

    # save the plot to file
    plt.tight_layout()
    plt.savefig(plotfname, dpi=plots["dpi"], ax=ax)
    plt.close()


# # AF added in function to plot visual gaze direction annotations onto the face images.
# def plot_visuals(folders, info, gaze, flmarksinds, plots):

#     # AF initiate logger
#     logger = logging.getLogger(__name__)

#     logger.info("PROCESSING VISUAL MARKERS ON FRAMES")

#     gcolours = {}
#     gcolours["left"] = "b"
#     gcolours["right"] = "r"
#     gcolours["middle"] = "w"
#     gcolours["blink"] = "w"
#     gcolours["outside"] = "g"
#     gcolours["unknown"] = "g"
#     eyelinewdt = 8

#     figsize = (20, 20)
#     dpi = 200
#     markersize = 2

#     videofolder = folders["frames"]
#     vfoldername = videofolder + info["study"]
#     vfolderlist = [
#         dirname for dirname in os.listdir(vfoldername) if not dirname[0] == "."
#     ]

#     # read the scoring file name mapping
#     infname = "respondent_info.xlsx"
#     insheet = "info"
#     infodset = pd.read_excel(infname, insheet, index_col=0, engine="openpyxl")
#     infodset = infodset.dropna(how="all")
#     infodset["respondent"] = infodset["respondent"].astype(int).astype(str)

#     scoringfolder = folders["spath"]
#     scorerlist = folders["scorers"]

#     for scorer in scorerlist:

#         for _, respinfo in infodset.iterrows():
#             # get the video folder
#             resp = respinfo["respondent"]
#             touse = respinfo["touse_" + info["study"].lower()]
#             nframesinfo = respinfo["nframes"]

#             if touse == 0:
#                 continue

#             foldername = folders["frames"] + info["study"] + "/" + resp + "/"
#             framelist = [
#                 fname
#                 for fname in os.listdir(foldername)
#                 if fname.endswith(info["imgextension"])
#             ]

#             if (nframesinfo == "all") or (np.isnan(nframesinfo)):
#                 nframes = len(framelist)
#             else:
#                 nframes = nframesinfo

#             if touse == 0:
#                 continue

#             logger.info(f"Processing {resp}, no. of frames:  {nframes}")

#             # vfoldername  = videofolder+study+"/"+vfolder+"/"
#             # framelist = [fname for fname in os.listdir(vfoldername) if fname.endswith(imgextension)]
#             # nframes = len(framelist)

#             # read the eye tracking processing
#             # procfolder = folders['proc']+study+'/'+vfolder+'/'
#             # etfname    = procfolder+vfolder+'_processed_extended.csv'
#             procfolder = folders["proc"] + \
#                 info["study"] + "/" + str(resp) + "/"
#             etfname = procfolder + str(resp) + "_processed_extended.csv"
#             if not os.path.isfile(etfname):
#                 logger.warning(f"ET file not available for {etfname}")
#                 continue

#             # read the eye tracking processing
#             etdset = pd.read_csv(
#                 etfname, index_col="frame", keep_default_na=True)
#             nframeset = etdset.shape[0]

#             scoring = {}

#             # read the scoring file
#             # sfoldername = scoringfolder+scorer+'/'+study+'/'+resp+'/'
#             # sfname = sfoldername+resp+'_scoring_processed.csv' # .format(rnumber)
#             sfoldername = (
#                 scoringfolder + scorer + "/" +
#                 info["study"] + "/" + str(resp) + "/"
#             )
#             sfname = (
#                 sfoldername + str(resp) + "_scoring_processed.csv"
#             )  # .format(rnumber)
#             if not os.path.isfile(sfname):
#                 logger.warning(f"Scoring file {sfname} does not exist")
#                 continue
#             scoring[scorer] = io_utils.read_scoring(sfname)

#             facefolder, facestatus = utils.create_folder(
#                 folders["output"] + "/aface/" +
#                 info["study"] + "/" + resp + "/"
#             )
#             if not facestatus:
#                 logger.warning(f"Folder cannot be created for {resp}: aborted")
#                 continue

#             for iframe, etrow in etdset.head(int(nframes)).iterrows():
#                 foldername = videofolder + "/" + \
#                     info["study"] + "/" + resp + "/"
#                 # fname = foldername + imgprefix + "{:05d}".format(iframe) + imgextension
#                 fname = (
#                     foldername
#                     + resp
#                     + "_"
#                     + info["imgprefix"]
#                     + "{:05d}".format(iframe)
#                     + info["imgextension"]
#                 )
#                 img = cv2.imread(fname)

#                 logger.info(f"frame: {iframe}")

#                 # convert image to gray
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                 (fx, fy, fw,
#                  fh) = etrow["fx"], etrow["fy"], etrow["fw"], etrow["fh"]
#                 if np.isnan(fx):
#                     # if no face detected save the image and continue
#                     fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1)
#                     ax.axis("off")
#                     ax.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
#                     plt.tight_layout()
#                     plt.savefig(
#                         "{:s}aface_{:03d}.png".format(facefolder, iframe),
#                         dpi=dpi,
#                         ax=ax,
#                     )
#                     plt.close()
#                     continue

#                 ebboxes = {}
#                 icircles = {}
#                 lmarksx = {}
#                 lmarksy = {}
#                 lmark = "eye"
#                 for eye in gaze["elist"]:
#                     ebboxes[eye] = (
#                         etrow["e" + eye[0] + "x"],
#                         etrow["e" + eye[0] + "y"],
#                         etrow["e" + eye[0] + "w"],
#                         etrow["e" + eye[0] + "h"],
#                     )
#                     icircles[eye] = (
#                         etrow["i" + eye[0] + "x"],
#                         etrow["i" + eye[0] + "y"],
#                         etrow["i" + eye[0] + "r"],
#                     )
#                     colnames = [
#                         eye[0] + lmark + str(ind) + "x"
#                         for ind in range(len(flmarksinds[eye[0] + lmark]))
#                     ]
#                     lmarksx[eye] = etrow[colnames].to_numpy(copy=True)
#                     lmarksx[eye] = np.hstack((lmarksx[eye], lmarksx[eye][0]))
#                     colnames = [
#                         eye[0] + lmark + str(ind) + "y"
#                         for ind in range(len(flmarksinds[eye[0] + lmark]))
#                     ]
#                     lmarksy[eye] = etrow[colnames].to_numpy(copy=True)
#                     lmarksy[eye] = np.hstack((lmarksy[eye], lmarksy[eye][0]))

#                 (fx, fy, fw, fh) = int(fx), int(fy), int(fw), int(fh)
#                 # draw the face and looks
#                 fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1)
#                 ax.axis("off")
#                 img = img[fy: fy + fh, fx: fx + fw]
#                 img2plot = img
#                 # draw the face image
#                 ax.imshow(cv2.cvtColor(img2plot, cv2.COLOR_GRAY2RGB))
#                 # draw the eye landmarks, pupil and looks

#                 for iscorer, scorer in enumerate(scorerlist):

#                     for eye in gaze["elist"]:
#                         (ex, ey, ew, eh) = ebboxes[eye]
#                         (ix, iy, ir) = icircles[eye]
#                         ax.plot(
#                             lmarksx[eye],
#                             lmarksy[eye],
#                             linewidth=eyelinewdt,
#                             color=gcolours[eye],
#                         )
#                         # ax.add_patch(mpl.patches.Rectangle((ex, ey), ew, eh, linewidth=eyelinewdt, edgecolor=colours[eye],facecolor='none') )
#                         ax.add_patch(
#                             mpl.patches.Circle(
#                                 (ex + ix, ey + iy),
#                                 ir,
#                                 linewidth=eyelinewdt,
#                                 edgecolor=gcolours[eye],
#                                 facecolor="none",
#                             )
#                         )

#                         # draw the horizontal look direction for eye
#                         shapes = {}
#                         shapes["shapemidx"] = ex + int(np.round(ew / 2))
#                         shapes["arrowlength"] = int(np.round(ew / 2))
#                         shapes["arrowleftx"] = shapes["shapemidx"] - int(
#                             np.round(shapes["arrowlength"] / 2)
#                         )
#                         shapes["arrowrightx"] = shapes["shapemidx"] + int(
#                             np.round(shapes["arrowlength"] / 2)
#                         )
#                         shapes["ellipsewidth"] = np.round(ew / 2)
#                         shapes["ellipseheight"] = np.round(eh / 2)

#                         if plots["do_scoring"]:
#                             # manual scoring

#                             shapes["arrowy"] = ey + (2 + iscorer) * eh
#                             hehdirval = scoring[scorer].loc[iframe,
#                                                             eye[0] + "eye"]
#                             hehdir = io_utils.get_direction(hehdirval)
#                             colour = gcolours[hehdir.lower()]
#                             plot_direction(hehdirval, colour, shapes, ax)

#                         # automatic scoring for individual eyes
#                         shapes["arrowy"] = ey + (4 + iscorer) * eh
#                         aehdirval = etrow["e" + eye[0] + "hdirval"]
#                         aehdir = io_utils.get_direction(aehdirval)
#                         colour = gcolours[aehdir.lower()]
#                         plot_direction(aehdirval, colour, shapes, ax)

#                     # overall gaze
#                     nosewidth = etrow["fnose3x"] - etrow["fnose0x"]
#                     shapes = {}
#                     shapes["shapemidx"] = etrow["fnose2x"]
#                     shapes["arrowlength"] = np.round(nosewidth / 2)
#                     shapes["arrowleftx"] = shapes["shapemidx"] - int(
#                         np.round(shapes["arrowlength"] / 2)
#                     )
#                     shapes["arrowrightx"] = shapes["shapemidx"] + int(
#                         np.round(shapes["arrowlength"] / 2)
#                     )
#                     shapes["ellipsewidth"] = np.round(nosewidth / 2)
#                     shapes["ellipseheight"] = np.round(ebboxes["left"][3] / 2)

#                     if plots["do_scoring"]:
#                         # manual scoring
#                         shapes["arrowy"] = np.mean(
#                             [ebboxes["left"][1], ebboxes["right"][1]]
#                         ) + (2 + len(scorerlist)) * np.mean(
#                             [ebboxes["left"][3], ebboxes["right"][3]]
#                         )
#                         hehdirval = scoring[scorer].loc[iframe, "overall"]
#                         hehdir = io_utils.get_direction(hehdirval)
#                         colour = gcolours[hehdir.lower()]
#                         plot_direction(hehdirval, colour, shapes, ax)

#                     # automatic scoring
#                     shapes["arrowy"] = np.mean(
#                         [ebboxes["left"][1], ebboxes["right"][1]]
#                     ) + (4 + len(scorerlist)) * np.mean(
#                         [ebboxes["left"][3], ebboxes["right"][3]]
#                     )
#                     aehdirval = etrow["hdirval"]
#                     aehdir = io_utils.get_direction(aehdirval)
#                     colour = gcolours[aehdir.lower()]
#                     plot_direction(aehdirval, colour, shapes, ax)

#                     plt.tight_layout()
#                     plt.savefig(
#                         "{:s}aface_{:03d}.png".format(facefolder, iframe),
#                         dpi=dpi,
#                         ax=ax,
#                     )
#                     plt.close()
#         #         break
#         #     break


def gaze_tcourse(etdset, eye='overall', columns=None, title='Timecourse', fname=None, plot_format="jpg"):

    
    dset = etdset.copy()
    
    PID_list = dset['PID'].unique()
    mean_length = []
    for p in PID_list:
        df = dset[dset['PID']==p]
        mean_length.append(len(df))

    mode = statistics.mode(mean_length)
    mode = mode - (mode*0.1)

    
    for p in PID_list:
        df = dset[dset['PID']==p]
        if len(df) < mode:
            # print(f'Participant {p} has been dropped due to difference from common framerate')
            dset.drop(dset.loc[dset['PID']==p].index, inplace=True)



    
    if eye == 'right':
        eye_index = 'argazeraw'
        score_index = 'srgazeraw'
    elif eye == 'left':
        eye_index = 'algazeraw'
        score_index = 'slgazeraw'
    else:
        eye_index = 'agazeraw'
        score_index = 'sgazeraw'
    
    statistic = "mean_cl_boot"
    na_rm = True
    erroralpha = 0.5
    shape = "."

    bgcolour = "white"
    dpi = 300
    axistitle_fontsize = 14
    axistext_fontsize = 12

    the_theme = p9.theme(
        panel_background=p9.element_rect(fill=bgcolour),
        axis_line_x=p9.element_line(size=0.5),
        axis_line_y=p9.element_line(size=0.5),
        axis_title=p9.element_text(size=axistitle_fontsize),
        axis_text=p9.element_text(size=axistext_fontsize),
        dpi=dpi,
        figure_size=(8, 2),
    )

    in_dset = pd.DataFrame({'PID': dset['PID'], 'frame': dset['frame']})
    value_vars = []

    if eye_index in dset:
        in_dset['automatic'] = dset[eye_index]
        value_vars.append('automatic')
    
    if score_index in dset:
        in_dset['manual'] = dset[score_index]
        value_vars.append('manual')

    if 'h_location' in dset:
        in_dset['timestamp'] = dset['h_location']
        value_vars.append('timestamp')



    vars2melt = ["PID", "frame"]
    var_name = "Scoring"
    value_name = "Gaze score"
    dset_long = pd.melt(
        in_dset,
        id_vars=vars2melt,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
    )


    colours = ["navy", "orange", "green"]
    yticks = [-1, 0, 1]
    ylabels = ["Right", "Middle", "Left"]
    hplot = (
        p9.ggplot(dset_long, p9.aes("frame", value_name, colour="Scoring"))
        + p9.geom_point(
            stat=p9.stat_summary(fun_data=statistic),
            shape=shape,
            alpha=erroralpha,
            na_rm=na_rm,
        )
        + p9.geom_errorbar(
            stat=p9.stat_summary(fun_data=statistic), alpha=erroralpha, na_rm=na_rm
        )
        #              + p9.geom_ribbon(stat=p9.stat_summary(fun_data=statistic))
        + p9.scale_color_manual(values=(colours))
        + p9.scale_y_continuous(breaks=yticks, labels=ylabels)
        + p9.ggtitle(title)
        + the_theme
    )


    if fname is None:
        hplot.draw()
    else:
        hplot.save(
            fname,
            format=plot_format,
        )

def eye_descriptives(dset, fname=None, plot_format="jpg"):

    axistitle_fontsize = 12
    axistext_fontsize = 12
    bgcolour = "white"
    dpi = 75
    plotformat = "jpg"

    statistic = "mean_cl_boot"

    the_theme = p9.theme(
        panel_background=p9.element_rect(fill=bgcolour),
        axis_line_x=p9.element_line(size=0.5),
        axis_line_y=p9.element_line(size=0.5),
        axis_title=p9.element_text(size=axistitle_fontsize),
        axis_text=p9.element_text(size=axistext_fontsize),
        dpi=dpi,
        figure_size=(8, 2),
    )

    # stats on width and height of eyes
    fdset2plot = dset.loc[:, ["frame", "elw",
                              "erw", "elh", "erh"]].copy().reset_index()
    fdset2plot.columns = [
        "participant",
        "frame",
        "Width left",
        "Width right",
        "Height left",
        "Height right",
    ]

    vars2melt = ["participant", "frame"]
    var_name = "Eye"
    value_name = "Measure (pixels)"
    value_vars = fdset2plot.columns[2:]
    fdset2plot_long = pd.melt(
        fdset2plot,
        id_vars=vars2melt,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
    )
    hplot = (
        p9.ggplot(fdset2plot_long, p9.aes(var_name, value_name))
        + p9.geom_boxplot()
        + p9.coord_flip()
        #          + p9.facet_ wrap("participant", nrow=5, ncol=5)
        + the_theme
        + p9.theme(axis_title_y=p9.element_text(color=bgcolour),
                   figure_size=(4, 4))
    )
    hplot.draw()

    if fname is not None:
        hplot.save(filename=fname, format=plot_format)


def descriptives_lkv(dset, fname=None, plot_format="jpg", save_csv=None):

    statistic = "mean_cl_boot"

    bgcolour = "white"
    dpi = 75
    axistitle_fontsize = 14
    axistext_fontsize = 12

    the_theme = p9.theme(
        panel_background=p9.element_rect(fill=bgcolour),
        axis_line_x=p9.element_line(size=0.5),
        axis_line_y=p9.element_line(size=0.5),
        axis_title=p9.element_text(size=axistitle_fontsize),
        axis_text=p9.element_text(size=axistext_fontsize),
        dpi=dpi,
        figure_size=(8, 2),
    )

    # # sanity check on source
    # print(dset.shape)
    # print(dset.loc[dset["agaze"] == 0, "asource"].value_counts())
    # print(dset.loc[dset["agaze"] == 1, "asource"].value_counts())
    # print(dset.loc[dset["agaze"] == 2, "asource"].value_counts())

    # length of lkv replacement
    lkvcols = [
        "participant",
        "allkvlen",
        "arlkvlen",
        "alkvlen",
        "sllkvlen",
        "srlkvlen",
        "slkvlen",
    ]
    lkvcols_names = [
        "participant",
        "Automatic L",
        "Automatic R",
        "Automatic",
        "Manual L",
        "Manual R",
        "Manual",
    ]
    dset2proc = dset.dropna(subset=lkvcols[1:], thresh=1)
    dset2proc.replace(to_replace=0, value=np.nan, inplace=True)
    print("Entire dataset:   ", dset.shape)
    print("Dataset with LKV: ", dset2proc.shape)

    if save_csv is not None:
        dset2proc[lkvcols].to_csv(save_csv)

    vars2melt = ["participant", "frame"]
    var_name = "Source"
    value_name = "LKV length"
    value_vars = lkvcols[1:]
    dset2proc_long = pd.melt(
        dset2proc,
        id_vars=vars2melt,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
    )
    for icolumn, column in enumerate(lkvcols):
        dset2proc_long[var_name].replace(
            to_replace=column, value=lkvcols_names[icolumn], inplace=True
        )
    the_theme = p9.theme(
        panel_background=p9.element_rect(fill=bgcolour),
        axis_line_x=p9.element_line(size=0.5),
        axis_line_y=p9.element_line(size=0.5),
        axis_title=p9.element_text(size=axistitle_fontsize),
        axis_text=p9.element_text(size=axistext_fontsize),
        dpi=dpi,
        figure_size=(8, 2),
    )

    hplot = (
        p9.ggplot(dset2proc_long, p9.aes(var_name, value_name))
        + p9.geom_boxplot()
        + p9.coord_flip()
        + the_theme
        + p9.theme(axis_title_y=p9.element_text(color=bgcolour),
                   figure_size=(4, 4))
    )
    hplot.draw()

    if fname is not None:
        hplot.save(filename=fname, format=plot_format)


def plot_ckap(ckap, save_path=None):

    plot_ckap = ckap[['PID', 'k_overall', 'k_left', 'k_right']]
    plot_low = ckap[['PID', 'CI_low_overall', 'CI_low_left', 'CI_low_right']]
    plot_upp = ckap[['PID', 'CI_upp_overall', 'CI_upp_left', 'CI_upp_right']]

    plot_ckap = plot_ckap.melt(id_vars="PID", value_name="k")
    plot_low = plot_low.melt(id_vars="PID", value_name="lb")
    plot_upp = plot_upp.melt(id_vars="PID", value_name="ub")

    plot_ckap = pd.concat([plot_ckap, plot_low['lb'], plot_upp['ub']], axis=1)
    

    plot = (p9.ggplot(plot_ckap)         # defining what data to use
    + aes(x = 'PID', y = 'k', color='variable')    # defining what variable to use
    + geom_pointrange(aes(ymin='lb', ymax='ub'), position = position_dodge(width = 0.5))
    + labs(y="Cohen's kappa score", x = "Participant ID.")
    + coord_cartesian(ylim=[-0,1])
    + scale_y_continuous(breaks=np.arange (0, 1.2, 0.2)) 
    + theme(axis_text_x = element_text(angle = 45, vjust = 1, hjust = 1))
    + theme(legend_position = "bottom", legend_title = element_blank(), legend_box_spacing=1) 
    + theme(panel_grid_major = element_blank(), panel_grid_minor = element_blank(), panel_background = element_blank(), axis_line = element_line(colour = "black"))
    + theme(figure_size=(16, 8))
    + geom_hline(yintercept=0.61, linetype="dashed", color = "black")
    + labs(y="Cohen's kappa score", x = "Participant ID.")
    )

    if save_path is not None:
        plot.save(filename=save_path)
    else:
        plot.draw()



