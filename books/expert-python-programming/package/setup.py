from setuptools import setup

setup(
    name = "mory-package",
    version = '0.0.1',
    description = 'mory package test',
    long_description = """very long description contains the whole details""",
    install_requires = [
        'pcn', 
    ],
    # trove classifiers
    classifiers = [
        'Development Status :: 4 - Beta',
        'Operating System :: OS independent'
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Deep Learning',
    ]
)


# basic usage: 
# python setup.py --help-commands