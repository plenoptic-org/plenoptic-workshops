# Welcome to plenoptic satellite event, VSS 2025

This site hosts the example notebook used for the plenoptic satellite event at VSS 2025. This three hour session aims to introduce the basics of using plenoptic in order to better understand computational visual models with simple examples. We hope to explain not just `plenoptic`'s syntax but also the type of reasoning that it facilitates.

The presentation I gave at the beginning of this session can be found [here](https://presentations.plenoptic.org/2025-05-16_vss-symposium/slides.html).

Before the satellite event, please try to follow the [setup](#setup) instructions below to install everything on your personal laptop.

## This website

This website contains rendered versions of the notebooks we will be working through during this workshop. During the workshop, attendees should look at the versions found under the `For users` section. These notebooks have some code pre-filled, as well as brief notes to help orient you. If you follow the setup instructions below, you will have editable copies of these notebooks on your laptop, and you are expected to follow along using these notebooks.

If you miss something or fall behind, you can look into the `For presenters` section, which includes the completed code blocks (along with some notes), so you can catch up.

After the workshop, we encourage you to return and check out the `Full notebooks` section, which, as the name implies, includes everything: explanatory text, code, and plots.

You may also find the [glossary](glossary.md) useful as you go through the notebook.

## Setup

:::{note}
If you would just like to install `plenoptic` to use it locally, follow [our installation instructions](https://plenoptic.readthedocs.io/en/latest/install.html). This tutorial contains some extra packages for this specific build.
:::

Before the event, please try to complete the following steps. If you are unable to do so, try to arrive to the event 30 minutes early so we can get it straightened out!

0. Make sure you have `git` installed. It is installed by default on most Mac and Linux machines, but you may need to install it if you are on Windows. [These instructions](https://github.com/git-guides/install-git) should help.
1. Clone the github repo for this workshop:
   ```shell
   git clone https://github.com/plenoptic-org/plenoptic-vss-2025.git
   ```

### Create a virtual environment with python 3.11

There are many ways to set up a python virtual environment. You can use your favorite way of doing so. If you don't have a preference or don't know what to do, choose one of the following:

:::::::{tab-set}
:sync-group: category

::::::{tab-item} uv
:sync: uv

:::::{tab-set}
:sync-group: os

::::{tab-item} Mac/Linux
:sync: posix

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) by running:
   ```shell
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   
2. Restart your terminal to make sure `uv` is available.
3. Install python 3.11:
   ```shell
   uv python install 3.11
   ```
   
4. Navigate to your cloned repo and create a new virtual environment:
   ```shell
   cd plenoptic-vss-2025
   uv venv -p 3.11
   ```
   
5. Activate your new virtual environment by running:
   ```shell
   source .venv/bin/activate
   ```
::::

::::{tab-item} Windows
:sync: windows

Open up `powershell`, then:

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. Install python 3.11:
   ```powershell
   uv python install 3.11
   ```
   
3. Navigate to your cloned repo and create a new virtual environment:
   ```powershell
   cd plenoptic-vss-2025
   uv venv -p 3.11
   ```
   
4. Activate your new virtual environment by running:
   ```powershell
   .venv\Scripts\activate
   ```

   :::{warning}
   You may receive an error saying "running scripts is disabled on this system". If so, run `Set-ExecutionPolicy -Scope CurrentUser` and enter `Unrestricted`, then press `Y`.
   
   You may have to do this every time you open powershell.
   
   :::

::::
:::::
::::::

::::::{tab-item} conda / miniforge
:sync: conda

1. Install [miniforge](https://github.com/conda-forge/miniforge) if you do not have some version of `conda` or `mamba` installed already.
2. Create the new virtual environment by running:
    ```shell
    conda create --name plenoptic-vss25 pip python=3.11 -c conda-forge
    ```
    Note the `-c conda-forge`!

3. Activate your new environment and navigate to the cloned repo: 
    ```shell
    conda activate plenoptic-vss25
    cd plenoptic-vss-2025
    ```
::::::

:::::::

#### Install dependencies and setup notebooks
    
1. Install the required dependencies. This will install pynapple and nemos, as well as jupyter and several other packages.
    ::::{tab-set}
    :sync-group: category
    
    :::{tab-item} uv
    :sync: uv

    ```shell
    uv pip install -r requirements.txt
    ```
    :::

    :::{tab-item} conda
    :sync: conda

    ```shell
    pip install -r requirements.txt
    ```
    :::
    ::::

2. Run our setup script to download data and prepare the notebooks:
    ```shell
    python scripts/setup.py
    ```
3. Confirm the installation and setup completed correctly by running:
    ```shell
    python scripts/check_setup.py
    ```

If `check_setup.py` tells you setup was successful, then you're good to go. Otherwise, please come to the satellite event room 30 minutes early on Monday, so we can get things going as quickly as possible.

After doing the above, the `notebooks/` directories within your local copy of the `plenoptic-vss-2025` repository will contain the jupyter notebooks for the event.

During the event, we will run through the notebooks in the order they're listed on this website. To open them, navigate to the `notebooks/` directory, activate your virtual environment and start `jupyter lab`:

::::::{tab-set}
:sync-group: category

:::::{tab-item} uv
:sync: uv

::::{tab-set}
:sync-group: os

:::{tab-item} Mac/Linux
:sync: posix

```shell
cd path/to/plenoptic-vss-2025/notebooks
source ../.venv/bin/activate
jupyter lab
```
:::

:::{tab-item} Windows
:sync: windows

```powershell
cd path\to\plenoptic-vss-2025\notebooks
..\.venv\Scripts\activate
jupyter lab
```
:::

:::::

:::::{tab-item} conda / miniforge
:sync: conda

```shell
cd path/to/plenoptic-vss-2025/notebooks
conda activate plenoptic-vss25
jupyter lab
```

:::::

::::::

:::{important}
You will also need `ffmpeg` installed in order to view the videos in the notebook. This is likely installed on your system already if you are on Linux or Mac (run `ffmpeg` in your command line to check). If not, you can install it via conda: `conda install -c conda-forge ffmpeg` or see their [install instructions](https://ffmpeg.org/download.html).

If you have `ffmpeg` installed and are still having issues, try running `conda update ffmpeg`.

:::

## Troubleshooting

- If you are on Mac and get an error related to `ruamel.yaml` (or `clang`) when running `pip install -r requirements.txt`, we think this can be fixed by updating your Xcode Command Line Tools.
- On Windows, you may receive an error saying "running scripts is disabled on this system" when trying to activate the virtual environment. If so, run `Set-ExecutionPolicy -Scope CurrentUser` and enter `Unrestricted`, then press `Y`. (You may have to do this every time you open powershell.)
- If you have multiple jupyter installs on your path (because e.g., because you have an existing jupyter installation in a conda environment and you then used `uv` to setup the virtual environment for this workshop), jupyter can get confused. (You can check if this is the case by running `which -a jupyter` on Mac / Linux.)
  To avoid this problem, either make sure you only have one virtual environment active (e.g., by running `conda deactivate`) or prepend `JUPYTER_DATA_DIR=$(realpath ..)/.venv/share/jupyter/` to your jupyter command above:

  ```shell
  JUPYTER_DATA_DIR=$(realpath ..)/.venv/share/jupyter/ jupyter lab
  ```

  (On Windows, replace `$(realpath ..)` with the path to the `ccn-software-jan-2025` directory.)
- We have noticed jupyter notebooks behaving a bit odd in Safari --- if you are running/editing jupyter in Safari and the behavior seems off (scrolling not smooth, lag between creation and display of cells), try a different browser. We've had better luck with Firefox or using the arrow keys to navigate between cells.
- On **Windows + conda**: if after installing conda the paths are not correctly set, you may encounter this error message: 
   ```
   conda : The term 'conda' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling of the name, or if a path was included, verify that the path is correct and try again.
   ```
  In this case, you can try the following steps:
  - Locate the path to the `condabin` folder. The path should look like: `some-folder-path\Miniforge3\condabin`. 
  
    The following powershell command could be useful (note that i am starting form C: as a root, but you can change that): 
    ```
    Get-ChildItem -Path C:\ -Directory -Recurse -Filter "condabin" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
    ```
  - Temporarily add conda to the paths: 
    ```
    $env:Path += ";some-folder-path\Miniforge3\condabin"
    ```
  - Initialize conda:
    ```
    conda init powershell 
    ```
  - Restart the powershell and check that conda is in the path. Run for example `conda --version`.
- If you see `sys:1: DeprecationWarning: Call to deprecated function (or staticmethod) _destroy.` when running `python scripts/setup.py`, we don't think this is actually a problem. As long as `check_setup.py` says everything looks good, you're fine!

## Binder

A binder instance (a virtual environment running on Flatiron's cluster) is provided in case we cannot get your installation working. To access it, click the "launch binder" button in the top left of this site or click [here](https://binder.flatironinstitute.org/v2/user/wbroderick/vss2025?labpath=notebooks).

You must login with the email address you provided when registering for the workshop. If you get a `403 Forbidden` error or would like to use a different email, send let Billy know.

- You are only allowed to have a single binder instance running at a time, so if you get the "already have an instance running error", go to the [binderhub page](https://binder.flatironinstitute.org/hub/hub/home) (or click on "check your currently running servers" on the right of the page) to join your running instance.
- If you lose connection halfway through the workshop, go to the [binderhub page](https://binder.flatironinstitute.org/hub/hub/home) to join your running instance rather than restarting the image.
- This is important because if you restart the image, **you will lose all data and progress**.
- The binder will be shutdown automatically after 1 day of inactivity or 7 days of total usage. Data will not persist after the binder instance shuts down, so **please download any notebooks** you want to keep.
- I will destroy this instance in 1 week, so that you can use it to play around during the conference. You can download your notebooks to keep them after the fact.

## Contents

See description above for an explanation of the difference between these two
notebooks.

```{toctree}
can_you_read.md
glossary.md
```

```{toctree}
:glob:
:caption: Full notebooks
:titlesonly:
full/*
```

```{toctree}
:glob:
:caption: For users (some code, some text)
:titlesonly:
users/*
```

```{toctree}
:glob:
:caption: For presenter reference (all code, no text)
:titlesonly:
presenters/*
```
