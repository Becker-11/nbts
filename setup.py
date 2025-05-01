# setup.py
from setuptools import setup, find_packages

setup(
    name="nbts",
    version="0.1.0",
    packages=find_packages(),          # this will pick up models/, solvers/, utils/, scripts/, test/
    install_requires=[
        "numpy", "scipy", "matplotlib",  # etc.
    ],
    entry_points={
        "console_scripts": [
            "sim = sim_heat_treatments:main",
            "const-sim = scripts.const_temp_sim_heat_treatments:main",
        ],
    },
)
