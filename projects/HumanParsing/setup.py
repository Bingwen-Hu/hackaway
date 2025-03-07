import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()


setuptools.setup(
    name = "psp",
    version = "0.1.0",
    author = "sirius demon",
    author_email = "mory2016@126.com",
    description="Pyramid Scene Parsing",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = "https://github.com/siriusdemon/pytorch-psp",
    packages=setuptools.find_packages(),
    package_data = {
        'rtpose': ['weights/PSPNet_last'],
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)