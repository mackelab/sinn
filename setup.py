
from setuptools import setup

setup(
      name='sinn',
      version='0.2.0dev',
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
                        'mackelab_toolbox',
                        'theano_shim',
                        'theano',
                        'tqdm',
                        'packaging',
                        'jsonpickle',
                        'odictliteral']
      # Add mackelab, theano as optional dependency
      # luigi is not included, since nothing imports luigi.py
      # A user importing sinn.models.luigi should expect to install luigi
     )
