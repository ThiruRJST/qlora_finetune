from setuptools import setup, find_packages
import os

# Read installation requirements from requirements.txt
def read_requirements():
    reqs = []
    with open('requirements.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                reqs.append(line)
    return reqs

setup(
    name='qlora_finetune',
    version='0.1',
    # Find packages inside the src directory if they contain an __init__.py file.
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            # Assumes that src/main.py defines a callable "main" function.
            'qlora_finetune=main:main'
        ]
    },
)