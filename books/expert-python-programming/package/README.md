## Mory package

If you want to use something different than reStructuredText markup language for
your project's README, you can still provide it as a project description on the PyPI
page in a readable form. The trick lies in using the pypandoc package to translate
your other markup language into reStructuredText while uploading the package to
Python Package Index. It is important to do it with a fallback to plain content of your
readme file, so the installation won't fail if the user has no pypandoc installed

```python
try:
    from pypandoc import convert

    def read_md(f):
        return convert(f, 'rst')
except ImportError:
    convert = None
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    def read_md(f):
        return open(f, 'r').read() # noqa`

README = os.path.join(os.path.dirname(__file__), 'README.md')
setup(
    name='some-package',
    long_description=read_md(README), 
    # ...
)
```