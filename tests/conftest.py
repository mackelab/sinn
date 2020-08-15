import pytest


import shutil
import os.path

@pytest.fixture
def clean_theano_dir(monkeypatch):
    """
    We have the problem that with testing, Theano seems to corrupt its own cache.
    I imagine that due to some race condition, multiple calls to `compile` end
    up using the same file name, which then leads to irreproducible errors.
    Since these don't occur when running any one test individually (with a
    clean cache), I think we can assume this is due to the testing environment.
    In any case it's not something we are equiped to fix.

    PROPER SOLUTION:
        Assign a different compilation directory for each test by setting
        THEANO_FLAGS="base_compiledir=~/.theano/pytest/[test_dir]"
        Unfortunately, `monkeypatch.setenv()` does not seem to work for this.

    WHAT WE ACTUALLY DID (Quick 'n dirty solution)
        At the beginning of each test using this fixture, the entire .theano
        directory is cleared. This is hacky for at least two reasons:
        - Any other files in .theano are also deleted.
          Are there long-lived files here ?
        - It hardcodes the location of the compile dir.
    """
    shutil.rmtree(os.path.expanduser("~/.theano"))
