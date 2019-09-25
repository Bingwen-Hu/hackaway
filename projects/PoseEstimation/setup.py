import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()


setuptools.setup(
    name = "rtpose",
    version = "0.1.0",
    author = "sirius demon",
    author_email = "mory2016@126.com",
    description="Multi-Person Pose Estimation",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = "https://github.com/siriusdemon/pytorch-rtpose",
    packages=setuptools.find_packages(),
    package_data = {
        'rtpose': ['weights/*.pth'],
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)