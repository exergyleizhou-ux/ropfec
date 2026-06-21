"""
setup.py stub for BOS-BMAC Phase 0 package.
For full packaging, use pyproject.toml or poetry.
"""
from setuptools import setup, find_packages

setup(
    name="bmac-engine",
    version="0.1.0",
    description="Phase 0 implementation of BOS-BMAC: ROP and FEC mapping to BOS platform per v1.0 spec.",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
    ],
    extras_require={
        "full": ["casadi>=3.6", "matplotlib"],
        "dev": ["pytest"],
    },
    python_requires=">=3.8",
    author="Grok (xAI) + user collaboration",
    url="https://example.com/bos-bmac",
)
