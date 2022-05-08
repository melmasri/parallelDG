#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import sys

from setuptools import setup, find_packages

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as f:
    readme = f.read()

packages = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
print("packages: {}".format(packages))

package_data = {
}

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
    requirements = f.readlines()

classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
]

setup(
    name='parallelDG',
    python_requires='<3.0',
    version=0.199,
    description='Parallel Bayesian structure learning in decomposable graphical models.',
    long_description=readme,
    packages=packages,
    package_data=package_data,
    install_requires=requirements,
    scripts=[
        "bin/parallelDG_ggm_sample",
        "bin/parallelDG_loglinear_sample",
        "bin/gen_g-intraclass_precmat",
        "bin/analyze_graph_trajectories",
        "bin/mh_ggm_sample",
        "bin/sample_g-inv_wish",
        "bin/sample_ggm_AR_data",
        "bin/sample_ggm_intraclass_data",
        "bin/sample_loglinear_data",
        "bin/sample_loglinear_parameters",
        "bin/sample_normal_data",
    ],
    author="Mohamad Elmasri",
    author_email='mohamad.elmasri@utoronto.ca',
    url='https://github.com/melmasri/parallelDG',
    download_url = '',
    license='Apache 2.0',
    classifiers=classifiers,
)
