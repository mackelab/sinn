
from setuptools import setup

setup(
      name='sinn',
      version='0.1.dev1',
      description="A package for Simulation and Inference of Neuron Networks",

      author="Alexandre René",
      author_email="alexandre.rene@caesar.de",

      license='MIT',

      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3 :: Only'
      ],

      packages=['sinn'],

      install_requires=['numpy >= 1.13',
                        'scipy',
                        'dill',
                        'tqdm']
      # Add mackelab, theano as optional dependency
     )
