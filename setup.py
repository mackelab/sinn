
from setuptools import setup

setup(
      name='sinn',
      version='0.1.0dev',
      description="A package for Simulation and Inference of Neuron Networks",

      author="Alexandre René",
      author_email="arene010@uottawa.ca",

      license='MIT',

      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3 :: Only',
          'Topic :: Scientific/Engineering :: Information Analysis'
      ],

      packages=['sinn'],

      install_requires=['numpy >= 1.13',
                        'scipy',
                        #'dill', # Histories are picklable now
                        'tqdm',
                        'packaging',
                        'jsonpickle']
      # Add mackelab, theano as optional dependency
      # luigi is not included, since nothing imports luigi.py
      # A user importing sinn.models.luigi should expect to install luigi
     )
