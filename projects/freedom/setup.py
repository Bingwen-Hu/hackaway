import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()


setuptools.setup(
    name = "freedom",
    version = "0.1.0",
    author = "sirius demon",
    author_email = "mory2016@126.com",
    description="Portable code for deep learning models",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = "https://github.com/siriusdemon/freedom",
    packages=setuptools.find_packages(),
    package_data = {
    },
    classifiers = [
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)