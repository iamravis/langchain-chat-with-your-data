from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT='-e .'

## Type hint 
def get_requirements(filepath:str) -> List[str]:
    ''' This function will return the list of requirements mentioned in the requirements file'''
    requirements = []
    
    with open(filepath) as f:
        requirements = f.readlines()
        requirements = [requirement.replace("\n","") for requirement in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup( 
    name='langchain-app', 
    version='0.1', 
    description='', 
    author='Ravi Shankar', 
    author_email='rsps1001@gmail.com', 
    packages= find_packages(where="./requirements.txt"), 
    install_requires = get_requirements('requirements.txt') 
)