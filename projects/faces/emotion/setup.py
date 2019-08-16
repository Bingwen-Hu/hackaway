import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()


setuptools.setup(
    name = "emotion",
    version = "0.0.1",
    author = "sirius demon",
    author_email = "mory2016@126.com",
    description="emotion classifier with dlib3",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = "https://github.com/siriusdemon/hackaway/tree/master/projects/emotion",
    packages=setuptools.find_packages(),
    package_data = {
        'emotion': ['emotion.hdf5'],
    },
    classifiers = [
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)