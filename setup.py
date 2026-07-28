"""Packaging metadata for the interference-index reference code."""

import os

from setuptools import find_packages, setup

NAME = "interference-index"
DESCRIPTION = "Interference index (xi): a scale-invariant metric for error cancellation in ML formation energy models."
URL = "https://github.com/KanzowA/InterferenceIndex"
EMAIL = "alexander.kanzow@stud.uni-goettingen.de"
AUTHOR = "Alexander Kanzow"
REQUIRES_PYTHON = ">=3.9.0"
VERSION = "1.0.0"

REQUIRED = [
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "torch",
    "pymatgen",
    "matminer",
]
EXTRAS = {
    "plotting": ["matplotlib", "prettytable"],
    "download": ["mp-api"],
}

HERE = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
        LONG_DESCRIPTION = "\n" + f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
