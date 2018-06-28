"""
IndicoAPI setup
"""
from sys import version_info
from setuptools import setup, find_packages

REQUIREMENTS = [
    "pandas>=0.23.1",
    "IndicoIo>=1.1.5",
    "enso>=0.1.6",
    "tqdm>=4.0.0",
    "numpy>=1.13.0",
    "scikit-learn>=0.18.0",
    "joblib>=0.11",
    "ftfy>=4.4.0",
    "spacy>=2.0.0",
    "msgpack-numpy>=0.4.1",
]
setup(
    name="finetune",
    packages=find_packages(),
    version="0.1.0",
    install_requires=REQUIREMENTS,
)
