import os
import sys
import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

requirements = []
with open('requirements.txt', 'r') as f:
    for line in f:
        requirements.append(line.strip())

#optional_requirements = []
#with open('optional-requirements.txt', 'r') as f:
#    for line in f:
#        optional_requirements.append(line.strip())

import subprocess

git_apt = "git+https://github.com/automl/Auto-PyTorch.git@nb301"

try:
    import autoPyTorch
except ImportError:
    if '--user' in sys.argv:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade',
                        '--user', git_apt], check=False)
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade',
                        git_apt], check=False)

setuptools.setup(
    name="nasbench301",
    version="0.3",
    author="AutoML Freiburg",
    author_email="zimmerl@informatik.uni-freiburg.de",
    description=("A surrogate benchmark for neural architecture search"),
    long_description=long_description,
    url="https://github.com/automl/nasbench301",
    long_description_content_type="text/markdown",
    license="3-clause BSD",
    keywords="machine learning"
             "optimization tuning neural architecture deep learning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
    ],
	python_requires='>=3',
    platforms=['Linux'],
    #install_requires=requirements[1:],
    include_package_data=True
#    extras_require=optional_requirements
)
