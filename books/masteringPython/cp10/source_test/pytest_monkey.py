import os


def test_chdir_monkeypatch(monkeypatch):
    monkeypatch.chdir('/Mory')
    assert os.getcwd() == '/Mory'
    monkeypatch.chdir('/')
    assert os.getcwd() == '/'


def test_chdir():
    original_directory = os.getcwd()
    try:
        os.chdir('/Mory')
        assert os.getcwd() == '/Mory'
        os.chdir('/')
        assert os.getcwd() == '/'
    finally:
        os.chdir(original_directory)

# Note: pytest monkey patch makes code clearer
