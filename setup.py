from setuptools import setup, find_packages
import os
import fnmatch

from cpunks import __version__


DATA_DIRS = [
    'cpunks/data',
    'cpunks/tutorial',
]

DATA_EXTENSIONS = [
    '*.pickle',
    '*.pkl',
    '*.json',
    '*.ipynb'
]


def get_data_files():
    data_files = []
    for data_dir in DATA_DIRS:
        matches = []
        for file in os.listdir(data_dir):
            for exc_pattern in DATA_EXTENSIONS:
                if fnmatch.fnmatch(file, exc_pattern):
                    dst_path = os.path.join(data_dir, file)
                    matches.append(dst_path)
                    break
        data_files.append((data_dir, matches))
    return data_files


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='cpunks-10k',
    version=__version__,
    author='Tnn1t1s',
    author_email='tnn1t1s@protonmail.com',
    description='CPUNKS-10K are subsets of the 10,000 labeled images in the CryptoPunks collection, organized & modified for use in Machine Learning research',
    long_description=long_description,
    # long_description_cont_type="text/markdown",
    url='http://github.com/tnn1t1s/cpunks-10k',
    license='BSD',
    packages=find_packages(),
    include_package_data=True,
    data_files=get_data_files(),
    scripts=['bin/cpunk'],
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'tensorflow',
        'jupyter',
    ],
    extras_require={
        'dev': [
            'flake8',
        ]
    },
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only', 
    ]
)
