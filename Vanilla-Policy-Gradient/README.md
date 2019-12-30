<h1 align="center"> Vanilla Policy Gradients </h1>

<h4 align="center"> Reimplementation of Vanilla Policy Gradients </h4>

<p align="center">
  <img src="https://img.shields.io/badge/Python-v3.6+-blue.svg">
  <img src="https://img.shields.io/badge/Pytorch-v1.3-orange.svg">
  <img src="https://img.shields.io/badge/Status-Complete-green.svg">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg">
</p>

<p align="center">
  <a href="#About">About</a> •
  <a href="#Requirements">Requirements</a> •
  <a href="#Algorithm">Algorithm</a> •
  <a href="#Environment">Environment</a> •
  <a href="#Training">Training</a> •
  <a href="#Results">Results</a> •
  <a href="#Sources">Sources</a>
</p>


## About
reimplementation of vanilla policy gradient applied to the CartPole-v0 environment from the OpenAI gym.

## Requirements

- Python 3.6
- Torch 1.1.0
- Numpy 1.16.4
- Matplotlib 2.2.4

## Algorithm

The vanilla policy gradient algorithm works by increasing the probability of actions that lead to the highest reward. This is done using both a policy and value network. The value network is trained using the observations and actions of an a single epoch. The policy network is trained using its actions from that episode as well as the corresponding advantage estimate that is calculated using the value network.

<p align="center">
  <img width="800" src="https://github.com/Gregory-Eales/ML-Reimplementations/blob/master/Vanilla-Policy-Gradient/img/vpg_pseudocode.png">
</p>


## Environment

<p align="center">
  <img width="460" height="300" src="https://github.com/Gregory-Eales/ML-Reimplementations/blob/master/Vanilla-Policy-Gradient/img/CartPole-v1.gif">
</p>

The enviroment used in this implementation is the CartPole-V0 enviroment provided by OpenAI gym. This environment provides a simple optimal controll challenge for an agent. The termination of an episode is triggered when the agent reaches 200 time steps, as well as when the agent looses control of the cartpole rig to a large degree.


<p align="center">
  <img width="800" src="https://github.com/Gregory-Eales/ML-Reimplementations/blob/master/Vanilla-Policy-Gradient/img/CartPoleTable.png">
</p>

## Training


## Results
By training the model over 1000 epochs with 4000 steps per epoch the agent was able to substantially improve its average reward nearly to the maximum obtainable reward. The agent also almost consisently made it to the maximum time step allowed inside the environment within 100 epochs.


<p align="center">
  <img height="450" src="https://github.com/Gregory-Eales/ML-Reimplementations/blob/master/Vanilla-Policy-Gradient/img/training_graph.gif">
</p>


## Sources

### Articles
* OpenAI SpinningUp (https://spinningup.openai.com/en/latest/algorithms/vpg.html)

### Papers
* Policy Gradient Methods for Reinforcement Learning with Function Approximation <br/>
  (https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)

## Meta

Gregory Eales – [@GregoryHamE](https://twitter.com/GregoryHamE) – gregory.hamilton.e@gmail.com

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/Gregory-Eales](https://github.com/Gregory-Eales)
