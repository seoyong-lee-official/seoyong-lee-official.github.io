---
layout: post
title: Diffusion Models
date: 2022-09-01 00:10:00-0400
description: Basics of diffusion models
tags: deep learning
related_posts: false
---

Foundation of Diffusion Model

## 1. Abstract
>
**Probabilistic generative model? **
- 주어진 $$data \ X_1, X_2, ..., X_N$$ 들이 어떠한 distribution $$p_{true}$$ 를 따르고 있다고 생각
- $$p_{\theta} \approx p_{true}$$ 가 되도록 NN 을 학습하여, $$p_{\theta}$$에서 새로운 data 들을 generate 하는 것이 목적
- $$e.g.,$$ $$VAE,GAN,Flow \ based..$$ 

>
** Diffusion Model? **
- probabilistic generative model 중 하나
- Forward process 와 Reverse process 로 구성됨
- Forward process는 data에 noise를 inject하며 data를 destruct하는 과정 (data $$\to$$ noise)
- Reverse process는 noise에서 data를 generate하는, forward process의 역과정(noise $$\to$$ data)

>
Forward process를 정의하고, 이에 해당하는 reverse process를 학습한다면, noise로부터 data($$e.g.image$$)를 generate할 수 있게 됨! 이게 끝
- Diffusion model의 forward process를 정의하는 방법은 다양함(그에 따라 이름이 다르게 붙음 : $$e.g.,DDPM, DDIM, ...$$)
- 가장 일반적인, forward process: **Stochastic Differential Equation(SDE)** 