#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Package setup
#
# ------------------------------------------------


# config
# ------
import re
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages


# config
# ------
class Config:
    def __init__(self, fi):
        with open(fi) as meta:
            for m in re.findall(r'(__[a-z]+__).*=.*[\'"](.+)[\'"]', meta.read()):
                setattr(self, m[0], m[1])
        return


config = Config('iso/__init__.py')


# requirements
# ------------
with open('requirements.txt', 'r') as reqs:
    requirements = list(map(lambda x: x.rstrip(), reqs.readlines()))

test_requirements = [
    'pytest',
    'pytest-cov',
    'pytest-runner'
]


# readme
# ------
with open('README.rst') as readme_file:
    readme = readme_file.read()


# exec
# ----
setup(
    name=config.__pkg__,
    version=config.__version__,
    description=config.__info__,
    long_description=readme,
    author=config.__author__,
    author_email=config.__email__,
    url=config.__url__,
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    keywords=['iso', 'machine-learning', 'learning', 'data', 'modelling', 'ai'],
    entry_points={
        'console_scripts': [
            'iso = iso.__main__:main'
        ]
    },
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    tests_require=test_requirements
)
