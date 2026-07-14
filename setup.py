'''
Defining python package, including metadata and dependencies, allowing everyone to install easily.

To install your package locally in editable mode,
use the following command in your terminal from the project’s root directory:

pip install -e .

This will install the package in your current folder (.) in editable mode (-e),
meaning any changes made to the code will be reflected without needing to reinstall the package. 🔄

Once the package is installed, you can use the prediction_function in other parts of the project like so:

from package_folder.climate import prediction_function

'''

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name='package_folder',
    version="0.0.1",
    install_requires=requirements,
    packages=find_packages()
)
