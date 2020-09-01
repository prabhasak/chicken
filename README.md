# The passive chicken and aggressive car problem

**Objective:** Inducing passive-aggressive behavior in self-driving cars using Inverse Reinforcement Learning (IRL), specifically Behavioral Cloning (BC). We use [Gym-Duckietown](https://github.com/duckietown/gym-duckietown), a self-driving car simulator for our experiments.

**Framework, langauge, OS:** Pytorch >= 0.4.1, Python 3.7, Windows 8.1

## Prerequisites
The Gym-Duckietown repo has been cloned into the "PedestrianSimulation" folder. Please follow the instructions on the repo to install all requirements

## Training
Add "PedestrianSimulation" to path in one of two ways:

    #From command line
    export PYTHONPATH="${PYTHONPATH}:`pwd`"

    #From ipython/python
    import sys
    sys.path.append('chicken/PedestrianSimulation') 

**Reinforcement learning:** code at PedestrianSimulation/learning/reinforcement/pytorch \
    python -m reinforcement.pytorch.train_reinforcement

**Imitation learning:** code at PedestrianSimulation/learning/imitation/pytorch \
    python -m imitation.pytorch.train_imitation

## Rendering Environment
To see the simulated pedestrain in the "PedestrianSimulation" folder \
    python manual_control.py

README last updated with instructions: Aug 2019