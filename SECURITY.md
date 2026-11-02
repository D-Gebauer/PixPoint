# Security policy

PixPoint is a scientific library that reads HEALPix maps and HDF5 files you
provide and writes HDF5 files. It opens no network connections and runs no code
from the files it reads. The realistic security concerns are therefore
malformed input files (which should raise a clean error, not crash the
interpreter) and the dependencies it pulls in (NumPy, SciPy, Numba, healpy,
h5py, optionally CuPy).

## Supported versions

Only the latest release on the `main` branch receives fixes.

## Reporting a vulnerability

If you believe you have found a security problem, please do not open a public
issue. Email the maintainer at git@gebauer.ai with a description and, if
possible, a minimal file or script that reproduces it. You will get an
acknowledgement within a week; fixes are released as a point release and noted
in the changelog.
