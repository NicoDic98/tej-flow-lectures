This repository contains materials for lectures on
**Normalizing Flows for Lattice Gauge Theory**
delivered at
_Frontiers and Challenges in Lattice Gauge Theory_, July 28 - August 1, 2025.

![flow sampling cartoon](./figs/flow_direct_sampling.jpg)

# Getting started
I suggest first setting up a local environment for developing code. You may
later want to upload your code to [Google Colab](https://colab.research.google.com)
to try running with a GPU.

### 1. Install Python
We will require Python >= 3.9.

> [!note]
> You can check if you already have an appropriate version of Python with
> `python -V`.

I suggest using `uv` because it is light and fast:
  
  * Install [uv](https://docs.astral.sh/uv/getting-started/installation/) in `~/.local/bin`
  ```
  curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh
  ```

You can add `~/.local/bin` to your default `$PATH` or just use the full path directly.
  
  * Install python
  ```
  uv python install 3.13
  ```

Python commands can now be run with `python3.13 ...`, but we will shortly set up
a virtual environment which will alias this to just `python` (see below). Other
options to install python include `conda` or directly installing from a package
manager or [python.org](http://python.org). I will also provide the non-uv
commands below.

### 2. Install packages
It is convenient to sandbox in a local directory using a virtual environment.

> [!warning]
> The installed packages require a little over 1GB of disk space. You may want
> to delete the `venv` directory when you are done with these exercises.
  
  * Create the virtual environment
  ```
  uv venv ./venv
  # OR: python -m venv ./venv
  ```

  * Activate and install into the virtual environment
  ```
  source ./venv/bin/activate
  uv pip install -r requirements.txt
  # OR: python -m pip install -r requirements.txt
  ```

  * Activate and confirm that you now have Pytorch
  ```
  python -c 'import torch; print(torch.version.__version__)'
  ```

Activate the environment to get access to the sandboxed python environment in a
new shell. You can also `deactivate` to exit the virutal env.


# Lectures and exercises
Interactive portions of lectures will involve developing some code in Jupyter
notebooks. Start up Jupyter Lab to follow along:
```
source ./venv/bin/activate
juypter-lab
```

Completed versions of the lecture notebooks are available with/without exercise
solutions (LectureX.ipynb and LectureXSolutions.ipynb), but I suggest
following along starting from LectureXSkeleton.ipynb to get more familiar with
the developed code.

Most of the exercises involve some small coding tasks. Think of them as a
starting point for your own exploration.
