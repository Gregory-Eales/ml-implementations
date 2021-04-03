<h1 align="center"> Proximal Policy Optimization </h1>

<h4 align="center"> Reimplementation of Proximal Policy Optimization </h4>

<p align="center">
  <img src="https://img.shields.io/badge/Python-v3.6+-blue.svg">
  <img src="https://img.shields.io/badge/Pytorch-v1.3-orange.svg">
  <img src="https://img.shields.io/badge/Status-Incomplete-red.svg">
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

## About:
Implementation of the proximal policy algorithm using pytorch and trained on the Pendulum-v0 environment provided by the OpenAI gym.

## Requirements:
- Pytorch
- Numpy
- Matplotlib
- TQDM

## Algorithm:
Proximal Policy Optimization is an on policy model free algorithm that can be used in both discrete and continuous environments. The implementation chosen is the ppo-clip variant as recommended in the OpenAI spinning up article. The clip variant uses clipping to incentivize the learning algorithm to stay near the original policy. This method of clipping ends up being much more simplistic in terms of implementation and at least as good as higher order penalization incurred in methods like TRPO.
<p align="center">
  <img width="624" height="394" src="https://github.com/Gregory-Eales/ML-Reimplementations/blob/master/Proximal-Policy-Optimization/img/ppo_pseudocode.png">
</p>


## Environment:
The Pendulum-V0 environment poses a classical control problem focused on the balancing of a pendulum using a single output representing joint output. This environment has an observation space of size 3, giving the x and y components of the pendulum relative to the horizontal and vertical planes, as well as the dot of the angle. The reward is given by normalizing the angle theta and ranges between -16.2736044 and 0. This means that the goal of the agent is to minimize the cost at any given time, resulting in a net reward of 0.

<p align="center">
  <img width="300" height="300" src="https://github.com/Gregory-Eales/ML-Reimplementations/blob/master/Proximal-Policy-Optimization/img/pendulum_v0.gif">
</p>

## Training:
- explain training methods
- plot accuracy and loss through training

## Results:
- show end result accuracy
- show prediction plot
- include closing thoughts + improvements

## Sources:

### Articles:
  - OpenAI Spinning Up (https://spinningup.openai.com/en/latest/algorithms/ppo.html)

### Papers:
  - Proximal Policy Optimization (https://arxiv.org/pdf/1707.06347.pdf)

## Meta:

Gregory Eales – [@GregoryHamE](https://twitter.com/GregoryHamE) – gregory.hamilton.e@gmail.com

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/Gregory-Eales](https://github.com/Gregory-Eales)
