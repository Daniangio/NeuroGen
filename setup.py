from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open(Path(__file__).parents[0] / "neurogen/_version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]
del version_dict

setup(
    name="neurogen",
    version=version,
    description="NeuroGen",
    download_url="https://github.com/Daniangio/NeuroGen",
    author="Daniele Angioletti",
    python_requires=">=3.10",
    packages=find_packages(include=["neurogen", "neurogen.*"]),
    entry_points={
        # make the scripts available as command line scripts
        "console_scripts": [
        ]
    },
    install_requires=[
        "numpy",
    ],
    zip_safe=True,
)
