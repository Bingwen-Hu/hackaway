## distribution
```sh
python3 setup.py sdist bdist_wheel
```

## upload To TestPypi
```sh
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

## upload to Pypi
```sh
twine upload dist/*
```

## install
```sh
pip install [name]
```
