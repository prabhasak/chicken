# IRL

Inverse Reinforcement Learning

## Rendering Environment

run python manual_control.py in PedestrianSimulation folder for viewing simulated pedestrain

## Learning
To Run the reinforcement training follow the commands


go to the following folder - /Users/yonarp/Projects/IRLTemp/gym-duckietown

    From command line
    export PYTHONPATH="${PYTHONPATH}:`pwd`"

    OR 

    From ipython/python
    import sys
    sys.path.append('/Users/yonarp/Projects/IRL/PedestrianSimulation') 


navigate to the learning folder - /Users/yonarp/Projects/IRLTemp/gym-duckietown/learning

Make sure your pytorch version is >= 0.4.1

For reinforcement learning

    python -m reinforcement.pytorch.train_reinforcement

For imitation learning

    make folder named models in /Users/yonarp/Projects/IRLTemp/gym-duckietown/learning/imitation/pytorch

    python -m imitation.pytorch.train_imitation
