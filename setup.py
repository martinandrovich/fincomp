#!/usr/bin/env python3
from setuptools import setup

setup(
	name="pyfin",
	version="1.0.0",
	description="",
	author="Martin Androvich",
	url="https://github.com/martinandrovich/pyfin",
	packages=["pyfin"],
	python_requires="==3.9.*",
	install_requires=[
		"numpy",
		"matplotlib",
	]
)
