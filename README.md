# BRM example code

The supplementary code has been seperated into two folders. The first folder, **video_processing**, contains a jupyter notebook and the Gazescorer package and demonstrates the video processing pipeline estimatinng the gaze orientation with three example vidoes. The second folder, **R_analysis**, contains two subfolders for the children and adult samples. In each folder is the full dataset used for the inter-rater reliability analysis and the R markdown file too run the analysis. 

## Video Processing - Gazescorer

Gazescorer is a python library that can be used for the analysis of videos to estimate the gaze orientation of participants. 

##### Installation

The simplest method to install the required packages to run Gazescorer is to create an anaconda environment using the provided enviroment file. Navigate to the **video_processing** in the terminal and run:

```sh
conda env create -f BRM_GS.yml
```

Once the environment has installed, activate it by running:
```sh
conda activate BRM_GS
```

If you would prefer to install the packages individually the required packages are below. These can generally be installed using either Conda or pip with the exception of FFmpeg and ffmpeg-python which are best to be installed using anaconda to ensure the correct version is installed. 

##### Dependencies:
- python=3.10
- cmake
    - Must be installed before dlib 
- dlib
- ipykernel
- ffmpeg v4.4.2 
    - For the correct version I recomend using:
        ```sh
        conda install -c anaconda ffmpeg
        ```
- ffmpeg-python
    - Install after ffmpeg to avoid conflicts. I would recomend installing using:
        ```sh
        conda install -c conda-forge ffmpeg-python
        ```
 - pandas
 - matplotlib
 - scipy
 - imutils
 - plotnine
 - sklearn
 - opencv-python
  

#### Running the analysis pipeline

To demonstrate the analysis pipeline three example videos have been provided. The analysis pipeline is in the jupyter notebook **Example_BRM.ipynb**. The first cell will process the three raw videos contained in the *input/example_video/* directory and append the scoring in the *input/example_scoring* directory. It will save the processed videos in the directory *input/example_vidoe/proccesed/*, the output data in the folder *output/example_datasets/*, and a plot of the detected face landmarks in the directory *output/face_lmarks/*.

The subsequent cells in the notebook will plot some simple inter-rater reliability outputs for both the static and dynamic phases. 

## R analysis

the dependecies for the R analysis are below:

##### dependecies

- dplyr 
- tidyr
- ggplot2
- reshape
- rel

Most of the packages can be installed using the install.packages() command. The *rel* package needs to be installed from an archived version. The first chunk in each of the R scripts can be used to install *rel* package. 





