# sphinx

## quickstart
```sh 
sphinx-quickstart
```
file `conf.py` contains all the settings.

## apidoc
to generate specific docs for some python source files, assuming they are in directory `src`, use this
```
sphinx-apidoc src -o docs
```
where `docs` is your directory that contains docs