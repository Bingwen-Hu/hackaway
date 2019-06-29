import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()


setuptools.setup(
    name = "insight",
    version = "0.1.0",
    author = "sirius demon",
    author_email = "mory2016@126.com",
    description="insight",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = "https://github.com/siriusdemon/hackaway/projects/insight",
    packages=setuptools.find_packages(),
    package_data = {
        'insight': ['model/*'],
        'insight': ['mtcnn-model/*'],
    },
    classifiers = [
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)