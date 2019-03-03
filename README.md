# Cart-Pole Q-Learning Reinforcer
This project contains several scripts used to build, train and showcase a (deep) neural network on a [cart-pole problem described by Barto, Sutton and Anderson](https://github.com/openai/gym/wiki/CartPole-v0).

## Models
All trained and untrained models are stored in the *models* directory by default. The name of the model is specified by the name of the subdirectory, which contains H5 files of untrained and trained model, and it's structure as an image.

## Model Builder
Included is also an easy to modify script *buildModel.py*, which builds a model with user-specified hidden layers, and a custom learning rate (feel free to change the parameters in, and passed to the builder).

## Agent
CartPoleAgent is a simple class that can load or save a specific model, decide upon the action, and remember and replay steps in the environment.

## CartPole Gym
The gym loads a specific model - either untrained or trained - as an agent and trains it on the cart-pole environment.

Features:
- higher reward when the cart is closer to the middle
- higher reward when the pole is upright
- optional: dynamic batch size and max agent memory size

## Showcase
It is also possible to showcase a specific model in the cart-pole environment without training it using a showcase script.