from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name = "DL_Project1",
    version = "0.0.1", 
    author = "Vimarish K M",
    author_email = "vimarish18100@gmail.com",
    package = find_packages(include=["Xray", "Xray.*"]),
    install_requires = get_requirements(r"C:\\Users\\vimar\\OneDrive\\Documents\\Projects\\dlproject1\\requirements_dev.txt"),
    
)