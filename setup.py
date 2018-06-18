"""
IndicoAPI setup
"""
from sys import version_info
from setuptools import setup, find_packages


setup(
    name="finetune",
    packages=find_packages(),
    setup_requires=open('requirements.txt').readlines(),
    version="0.1.0",
)
