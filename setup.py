"""Setup for hybrid_gait package."""

import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
    name="hybrid_gait",
    version="0.0.1",
    author="Derek Wang",
    author_email="416338223@qq.com",
    description="Trainning hybrid gait for mini cheetah.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Derek-TH-Wang/hybrid_gait.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)

