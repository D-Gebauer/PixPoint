# Installation

PixPoint needs Python 3.12 or newer and runs on Linux and macOS. Its
dependencies are NumPy, SciPy, Numba, healpy, h5py and tqdm, all of which pip
installs for you.

## From PyPI

```bash
pip install pixpoint
```

## From GitHub

The development version:

```bash
pip install git+https://github.com/D-Gebauer/PixPoint.git
```

Or, from a clone, in editable mode with the development tools:

```bash
git clone https://github.com/D-Gebauer/PixPoint.git
cd PixPoint
pip install -e ".[dev]"
```

## GPU support (optional)

Everything in PixPoint works on the CPU. To run on an NVIDIA GPU, install
[CuPy](https://docs.cupy.dev/en/stable/install.html) for your CUDA version in
the same environment, for example

```bash
pip install "pixpoint[gpu]"     # PixPoint + cupy-cuda12x
pip install cupy-cuda12x        # CUDA 12 (what the gpu extra installs)
pip install cupy-cuda13x        # CUDA 13
```

The CUDA 12 build runs on any driver that supports CUDA 12 or newer. Install
only one of the CuPy packages, and not the plain `cupy` package, which builds
CuPy from source.

PixPoint's GPU kernels are compiled the first time they are used, by the CUDA
runtime compiler that ships with CuPy; no compiler or CUDA toolkit installation
of your own is needed, but the driver must support CUDA 12 or newer. The first
measurement after importing therefore takes a few seconds longer than the rest.

With CuPy installed, `device="auto"` (the default everywhere) picks the GPU;
`device="cpu"` forces the CPU, `device=0` a particular GPU, and a list such as
`device=[0, 1]` spreads a measurement over several GPUs (see
[Running on GPUs](performance.md)).

## Checking the installation

```python
import pixpoint
print(pixpoint.__version__)

from pixpoint.backend import get_backend
print(get_backend("auto").name)      # "cupy" on a GPU, "numpy" on the CPU
```

To run the test suite from a clone (about one minute for the fast part):

```bash
pytest -m "not gpu and not slow"
```
