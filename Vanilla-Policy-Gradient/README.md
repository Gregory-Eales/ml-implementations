# Vanilla-Policy-Gradient
implementation of vpg

## About
reimplementation of vanilla policy gradient applied to the CartPole-v0 environment from the OpenAI gym<br/>

## Model

The vanilla policy gradient algorithm is a very straight forward rl process that increases the probability of getting higher rewards and decreasing the probability of getting lower rewards...

## Environment

<p align="center">
  <img width="460" height="300" src="https://github.com/Gregory-Eales/ML-Reimplementations/blob/master/Vanilla-Policy-Gradient/img/CartPole-v1.gif">
</p>

The enviroment used in this implementation is the CartPole-V1 enviroment provided by OpenAI gym. This environment provides a simple...

<p align="center">
<table>
<tr><th> Observations </th><th> Actions </th></tr>
<tr><td>

Num | Observation | Min | Max
---|---|---|---
0 | Cart Position | -2.4 | 2.4
1 | Cart Velocity | -Inf | Inf
2 | Pole Angle | ~ -41.8&deg; | ~ 41.8&deg;
3 | Pole Velocity At Tip | -Inf | Inf

</td><td>

Num | Action
--- | ---
0 | Push cart to the left
1 | Push cart to the right

</td></tr> </table>
</p>

## Training

By training the model over...

## Results

Given the performance...

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
