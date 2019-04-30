import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()


setuptools.setup(
    name = "landmark",
    version = "0.1.0",
    author = "sirius demon",
    author_email = "mory2016@126.com",
    description="landmark detector with dlib3",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = "https://github.com/siriusdemon/hackaway/tree/master/projects/landmark",
    packages=setuptools.find_packages(),
    package_data = {
        'landmark': ['sd_landmark.pth'],
    },
    classifiers = [
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)