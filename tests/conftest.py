import pytest


import shutil
import os.path
import tempfile

tmpstore = {}

@pytest.fixture
def clean_theano_dir(monkeypatch):
    """
    We have the problem that with testing, Theano seems to corrupt its own cache.
    I imagine that due to some race condition, multiple calls to `compile` end
    up using the same file name, which then leads to irreproducible errors.
    Since these don't occur when running any one test individually (with a
    clean cache), I think we can assume this is due to the testing environment.
    In any case it's not something we are equiped to fix.

    SOLUTION: Assign a different compilation directory for each test.
    """
    import theano
    tmpdir = tempfile.TemporaryDirectory()
    theano.config.compiledir = tmpdir.name
    # To keep the temporary directory open while the test runs, we assign
    # the object to the global variable 'tmpstore' which is kept alive.
    # When the test is finished, monkeypatch undoes this modification, thus
    # destroying `tmpdir`, and the associated directory is deleted
    monkeypatch.setitem(tmpstore, 'compiledir', tmpdir)

## 'slow' mark & '--runslow' option, following the example from the pytest docs: https://docs.pytest.org/en/6.2.x/example/simple.html?highlight=fixtures#control-skipping-of-tests-according-to-command-line-option

def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runslow-only", action="store_true", default=False, help="only run slow tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    if config.getoption("--runslow-only"):
        # --runslow-only given in cli: skip all non-slow tests
        skip_fast = pytest.mark.skip(reason="--runslow-only option skips all fast tests")
        for item in items:
            if "slow" not in item.keywords:
                item.add_marker(skip_fast)
    else:
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
