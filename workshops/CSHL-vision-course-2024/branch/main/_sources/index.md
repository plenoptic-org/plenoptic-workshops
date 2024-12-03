# Welcome to plenoptic tutorial, CSHL Vision Course 2024

This site hosts the example notebook used for the plenoptic tutorial given at the Cold Spring Harbor Labs Computational Neuroscience: Vision course in July 2024. This one-hour(ish) tutorial aims to introduce the basics of using plenoptic in order to better understand computational visual models with simple examples. We hope to explain not just `plenoptic`'s syntax but also the type of reasoning that it facilitates.

The presentation I gave at the beginning of this session can be found [here](https://labforcomputationalvision.github.io/plenoptic_presentations/2024-07-12_CSHL/slides.html).

This website contains two versions of the notebook we'll use today: [with](introduction.md) and [without](introduction-stripped.md) explanatory text. Today you'll run the version without explanatory text, which contains cells of code, while listening to my description. If later you wish to revisit this material, the version with explanatory text should help you.

You may also find the [glossary](glossary.md) useful as you go through the notebook.

You can also [follow the setup instructions here](#setup) to download these notebooks and run them locally, but to avoid potential installation issues in this brief period of time, we'll use binder instead. Click on the `launch binder` badge on the upper left sidebar, which will then prompt you to login. Use the google account that you gave to the class organizers; if you get a 403 forbidden error or would like to use a different account, let me know so that I can give it permission. The binder instance provides a GPU with the environment necessary to run the notebook. See [the section below](#binder) for more details on the binder, including some important usage notes.

## Setup

:::{note}
If you would just like to install `plenoptic` to use it locally, follow [our installation instructions](https://plenoptic.readthedocs.io/en/latest/install.html). This tutorial contains some extra packages for this specific build.
:::

While we'll use the binder during this tutorial, if you'd like to run the notebooks locally, you'll need to set up a local environment. To do so: 

0. Make sure you have `git` installed. It is installed by default on most Mac and Linux machines, but you may need to install it if you are on Windows. [These instructions](https://github.com/git-guides/install-git) should help.
1. Clone the github repo for this tutorial:
   ```shell
   git clone https://github.com/plenoptic-org/plenoptic-cshl-vision-2024.git
   ```
2. Create a new python 3.11 virtual environment. If you do not have a preferred way of managing your python virtual environments, we recommend [miniconda](https://docs.anaconda.com/free/miniconda/). After installing it (if you have not done so already), run 
    ```shell
    conda create --name cshl2024 pip python=3.11
    ```
3. Activate your new environment:
    ```shell
    conda activate cshl2024
    ```
4. Navigate to the cloned github repo and install the required dependencies.
    ```shell
    cd plenoptic-cshl-vision-2024
    pip install -r requirements.txt
    ```

    :::{important}
    You will also need `ffmpeg` installed in order to view the videos in the notebook. This is likely installed on your system already if you are on Linux or Mac (run `ffmpeg` in your command line to check). If not, you can install it via conda: `conda install -c conda-forge ffmpeg` or see their [install instructions](https://ffmpeg.org/download.html).
    
    If you have `ffmpeg` installed and are still having issues, try running `conda update ffmpeg`.
    
    :::
    
5. Run the setup script to prepare the notebook:
   ```shell
   python scripts/setup.py
   ```
   
   :::{important}
   It's possible this step will fail (especially if you are on Windows). If so, go to the [notebook on this site](introduction-stripped.md) and download it manually.
   :::

6. Open up jupyter, then double-click on the `introduction-stripped.ipynb` notebook:
   ```shell
   jupyter lab
   ```

## Binder

Some usage notes:

- You are only allowed to have a single binder instance running at a time, so if you get the "already have an instance running error", go to the [binderhub page](https://binder.flatironinstitute.org/hub/hub/home) (or click on "check your currently running servers" on the right of the page) to join your running instance.
- If you lose connection halfway through the workshop, go to the [binderhub page](https://binder.flatironinstitute.org/hub/hub/home) to join your running instance rather than restarting the image.
- This is important because if you restart the image, **you will lose all data and progress**.
- The binder will be shutdown automatically after 1 day of inactivity or 7 days of total usage. Data will not persist after the binder instance shuts down, so **please download any notebooks** you want to keep.
- I will destroy this instance in 2 weeks, so that you can use it to play around during the course. You can download your notebooks to keep them after the fact. If you do so, see the [setup instructions](#setup) for how to create the environment for running them locally, and let me know if you have any problems!

## Contents

See description above for an explanation of the difference between these two
notebooks.

```{toctree}
glossary.md
introduction.md
introduction-stripped.md
```
