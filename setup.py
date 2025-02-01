from setuptools import setup, find_packages
from typing import List

HYPHEN_DOT_E = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requiements = []
    with open(file_path, 'r') as file_obj:
        requiements = file_obj.readlines()
        requiements = [req.strip() for req in requiements]

    if HYPHEN_DOT_E in requiements:
        requiements.remove(HYPHEN_DOT_E)

    return requiements


setup(
    name='Classification',
    version='0.0.1',
    author='Harsh',
    author_email='harshpatel4877@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)