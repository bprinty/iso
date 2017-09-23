#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# package setup
#
# @author <bprinty@gmail.com>
# ------------------------------------------------


# config
# ------
import jade
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


# requirements
# ------------
with open('requirements.txt', 'r') as reqs:
    requirements = map(lambda x: x.rstrip(), reqs.readlines())

test_requirements = [
    'pytest',
    'pytest-runner'
]


# files
# -----
with open('README.rst') as readme_file:
    readme = readme_file.read()


# exec
# ----
setup(
    name='jade',
    version=jade.__version__,
    description='Package for managing data transformations in complex machine-learning workflows.',
    long_description=readme,
    author='Blake Printy',
    author_email='bprinty@gmail.com',
    url='https://github.com/bprinty/jade.git',
    packages=['jade'],
    package_data={'jade': 'jade'},
    entry_points={
        'console_scripts': [
            'jade = jade.__main__:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license='Apache-3.0',
    zip_safe=False,
    keywords=['jade', 'machine-learning', 'learning', 'data', 'modelling', 'ai'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache-3.0 License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    tests_require=test_requirements
)
