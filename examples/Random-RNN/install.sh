#! /bin/bash

set -e

ENVNAME="sinn_random-rnn"
ENVYML="random-rnn.yaml"
ENVDIR="../env"

# Change to the script's directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Switching to directory $DIR"
cd "$DIR"

# Get the list of current conda environments
envlist="$(conda env list)"

if [[ "$envlist" == *"$ENVNAME"* ]]; then
  echo "Environment '$ENVNAME' is already installed."
  exit
fi

# Make ENVDIR absolute
ENVDIR="$(readlink -f "$ENVDIR")"

# Make `conda activate` work in shell script  (https://github.com/conda/conda/issues/7980#issuecomment-492784093)
eval "$(conda shell.bash hook)"

# Create and activate new environment
conda env create --prefix "$ENVDIR/$ENVNAME" --file "$ENVYML"
conda activate "$ENVDIR/$ENVNAME"

python -m ipykernel install --user --name "$ENVNAME" --display-name "Python ($ENVNAME)"


# Get the list of current conda environment directories
envdirs="$(conda config --show envs_dirs)"
if [[ ! "$envdirs" == *"$ENVDIR"* ]]; then
  # The environment parent directory is not yet known to conda; add it
  conda config --append envs_dirs "$ENVDIR"
fi

# Wrap-up: fix Theano conda installation

# # Download scan_perform.c by hand because it is missing from the conda distribution (https://github.com/Theano/Theano/issues/6753)
# (v1.0.4 only)
# cd "$ENVDIR/$ENVNAME/lib/python3.8/site-packages/theano/scan_module/"
# mkdir c_code
# cd c_code
# wget https://raw.githubusercontent.com/Theano/Theano/e0167f24ae896a2e956cdd99a629910cd717a299/theano/scan_module/c_code/scan_perform.c
