# -*- coding: utf-8 -*-
"""
Created Fri Mar 10 2017

author: Alexandre René
"""

import signal
import logging

logger = logging.getLogger('sinn.process_handling')

# http://stackoverflow.com/a/22348885
class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def get_ipp_client(ipp_profile=None, ipp_url_file=None, wait=2):
    """Tries to create an IPyParallel Client and connect it to a running process.
    Returns the Client instance. If no process is found, returns None.
    At least one of `ipp_profile` or `ipp_url_file` should be specified; priority
    is given to `ipp_url_file` if both are given; the other is ignored.

    Parameters
    ----------
    ipp_profile: str
        IPython profile name, where one would find the ipcontroller.json file

    ipp_url_file: str
        Path to the ipcontroller.json file

    wait: float | int
        Number of seconds to try connecting to an IPyParallel controller.
    """
    try:
        import ipyparallel as ipp  # Only require this dependency if this function is called
    except ImportError:
        return None # IPyParallel is not installed, so there's definitely no process running

    # timeout ipp.Client after 2 seconds
    if ipp_url_file is not None:
        try:
            with timeout(2):
                ippclient = ipp.Client(url_file=ipp_url_file)
        except TimeoutError:
            logger.info("Unable to connect to ipyparallel controller.")
            ippclient = None
        else:
            logger.info("Connected to ipyparallel controller.")
    elif ipp_profile is not None:
        try:
            with timeout(2):
                ippclient = ipp.Client(profile=ipp_profile)
        except TimeoutError:
            logger.info("Unable to connect to ipyparallel controller with "
                        "profile '" + ipp_profile + ".'")
            ippclient = None
        else:
            logger.info("Connected to ipyparallel controller for profile '" + ipp_profile + "'.")
    else:
        ippclient = None

    return ippclient
