# Getting started
I suggest first setting up a local environment for developing code. You may
later want to upload your code to [Google Colab](https://colab.research.google.com)
to try running with a GPU.

### Install Python
We will require Python >= 3.9.

> [!note]
> You can check if you already have an appropriate version of Python with
> `python -V`.

I suggest using `uv` because it is light and fast:
  
  * Get [uv](https://docs.astral.sh/uv/getting-started/installation/)
  ```
  curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh
  ```
  
  * Install python
  ```
  uv python install 3.13
  ```

Other options include `conda` or directly installing from a package manager or
python.org.

### Install packages
It is convenient to sandbox in a local directory:

> [!warning]
> The installed packages require a little over 1GB of disk space. You may want
> to delete the `venv` directory when you are done with these exercises.
  
  * Create the virtual environment
  ```
  python -m venv ./venv
  
  ```

  * Activate and install into the virtual environment
  ```
  source ./venv/bin/activate
  python -m pip install -r requirements.txt
  ```

  * Confirm that you now have Pytorch
  ```
  python -c 'import torch; print(torch.version.__version__)'
  ```
