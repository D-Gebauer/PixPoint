"""The CUDA kernel sources must be pure ASCII.

PixPoint reads them as UTF-8, but CuPy 13 re-encodes the source as ASCII
when it compiles a RawModule, so a single non-ASCII character (even in a
comment) makes every kernel of that file fail to compile there (seen on
Perlmutter, CuPy 13.4.1, with PixPoint 8.0.0).
"""

from pathlib import Path

import pytest

import pixpoint

CUDA_DIR = Path(pixpoint.__file__).parent / "cuda"
SOURCES = sorted(CUDA_DIR.glob("*.cu")) + sorted(CUDA_DIR.glob("*.cuh"))


def test_sources_found():
    assert SOURCES, f"no CUDA sources in {CUDA_DIR}"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_cuda_source_is_ascii(path):
    text = path.read_text(encoding="utf-8")
    bad = [
        (line_no, line.strip())
        for line_no, line in enumerate(text.splitlines(), 1)
        if not line.isascii()
    ]
    assert not bad, f"non-ASCII characters in {path.name}: {bad[:5]}"
