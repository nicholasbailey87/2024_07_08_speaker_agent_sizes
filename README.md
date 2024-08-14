# Approximate speaker agent sizes used in emergent language research

## About this repository

This repository represents our reasonable efforts to run the code provided in some of the studies included in the survey paper "Deep Learning Agents and the Emergence of Compositional
Languages: Approaches, Inductive Biases and Measurement".

Our motivation was to substantiate the claim that most recent research into compositional emergent languages uses relatively small deep learning models, and indeed the apparent outcome is that most of the included studies use a speaker agent that is smaller than the 4.2m-parameter MobileNet v1, a model developed by Google in 2017 intended to be small enough to work on a mobile device.

The approach to including other people's code has simply been to copy all the files in their repository into a folder in this repository. We have not used git submodules.

In many cases, we have changed the provided code in order to make it easier to instantiate a speaker agent inside a Jupyter notebook. The changes tracked in this git repository will reveal where and how we have done this for those who are curious.

The results here are approximate as, despite paying careful attention to the provided code and referring to the related papers where necessary, running other people's code is a difficult business and we may have made some mistakes.

If you spot a mistake in the way we are running a study that you were involved with, please don't hesitate to contact `nicholas.bailey` `@` `city.ac.uk`

## Getting started instructions for Windows

(Hopefully these are clear enough that you can adapt them to your favourite OS if it's not Windows)

To prepare an environment to run agent_sizes.ipynb, first install miniconda and create a conda environment with Python v3.8 in the terminal:

```
conda create --name agent_sizes python=3.8
```

Activate the environment

```
conda activate agent_sizes
```

Then run the following commands to install requirements:

```
conda install -y --file conda_requirements.txt -c conda-forge -c nvidia -c pytorch
pip install -r pip_requirements.txt
```

You can then open Jupyter Lab with

```
jupyter lab
```

In the **Kernel** menu, choose **Restart kernel and run all cells** and this should build a dataset called output.csv

The process can take a while, but Jupyter Lab should indicate that it is still working.