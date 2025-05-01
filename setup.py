# setup.py
from setuptools import setup, find_packages

setup(
    name="nbts",
    version="0.1.0",
    packages=find_packages(),         
    install_requires=[
        "numpy", "scipy", "matplotlib",  # etc. need to be added 
    ],
    entry_points={
        "console_scripts": [
            "sim = sim_heat_treatments:main",        ],
    },
)
