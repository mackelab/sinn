
from setuptools import setup

setup(
      name='sinn',
      version='0.2.0rc1',
      description="A package for Simulation and Inference of Neuron Networks",

      author="Alexandre René",
      author_email="arene010@uottawa.ca",

      license='MIT',

      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3 :: Only',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Topic :: Scientific/Engineering :: Information Analysis'
      ],

      packages=['sinn'],

      install_requires=['numpy >= 1.13',
                        'scipy',
                        'pydantic',
                        'mackelab-toolbox[iotools,utils,typing,parameters,theano] >= 0.2.0a1',
                        'theano_shim >= 0.2.3',
                        'theano-pymc',
                        'tqdm',
                        'packaging',
                        'odictliteral'],
      # Add theano as optional dependency

      extras_require = {
        'dev': [
            'pytest',
            'pytest-xdist'
        ]
      }
     )
