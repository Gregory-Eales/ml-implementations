<h1 align="center"> Deep Deterministic Policy Gradients </h1>

<h4 align="center"> Reimplementation of the Deep Deterministic Policy Gradient algorithm </h4>

<p align="center">
  <img src="https://img.shields.io/badge/Python-v3.6+-blue.svg">
  <img src="https://img.shields.io/badge/Dependency-v1.3-orange.svg">
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

## About:
implementation of the DPPG algorithm applied to the LunarLanard-v2 gym environment using pytorch

## Requirements:
- Pytorch
- Numpy
- Gym
- Box-2D
- TQDM

## Algorithm:
This algorithm learns the Q function of an environment while also learning policy whos goal is to maximize Q. This algorithm employs a replay buffer which it uses to learn off policy. It is based on the idea that if you know a Q funtion that is close to the optimal Q then you know which actions are approximately optimal, effectivly solving the environment.

<p align="center">
  <img width=400 src="img/DDPG-Psuedocode.png">
</p>

## Environment:
The environment used in this implementation was the LunarLander-v2 environment from the OpenAI gym. The algorithm requires the continuous variant and requires the agent to move

<p align="center">
  <img width=400 src="img/Lunar-Lander-Example.gif">
</p>

## Training:
The model was trained with a 3 layer MLP for all of the Q and Policy networks using ADAM for faster optimization.

- show training graph

## Results:
- show end result accuracy 
- show prediction plot
- include closing thoughts + improvements

## Sources:
- [Lunar Lander Example gif](https://stable-baselines.readthedocs.io/en/master/guide/examples.html)
- [Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
- [DDPG Psuedocode](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

## Meta:

Gregory Eales – [@GregoryHamE](https://twitter.com/GregoryHamE) – gregory.hamilton.e@gmail.com

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/Gregory-Eales](https://github.com/Gregory-Eales)



