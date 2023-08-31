---
layout: post
title: Diffusion Models
date: 2023-08-31 15:37:00
description: basics of diffusion models
tags: DL
categories: sample-posts
related_posts: false
---

Foundation of Diffusion Model
_Ernest Ryu 교수님, Mathematical Algorithms 2 수업내용 기반_

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

## 2. SDE formulation
### 2-1. Forward process
We consider the following SDE in diffusion models.
>
$$dX_t = f(X_t, t)dt + g(t)dW_t$
- $$X_t,f \in \mathbb{R}^d, g \in \mathbb{R}^{d \times d}$
- $W_t$ : d independent 1 dimensional Wiener process


Wiener process와 SDE를 몰라도, 다음과 같이 SDE를 생각 가능
>
$X_{k+1} = X_{k} + \Delta{t}f(X_k , k \Delta t) + g(k \Delta t)\sqrt{\Delta t}Z_k$
- $Z_k \sim N(0,I)$  : d dimensional Gaussain r.v.  

즉 SDE를 forward process로 정했다는 것은, $X_0, \ f, \ g, \ Z_0$ 을 바탕으로 $X_1$을 계산하고,  $X_1, \ f, \ g, \ Z_1$ 을 바탕으로 $X_2$를 계산하고, ... , 이 과정을 반복하여 $X_0 , X_1, ..., X_T$를 얻는것.
$X_0$를 일반적인 image로 생각하면 되고, 여기에 noise $Z_i$가 계속해서 추가되므로 $X_T$에 갈수록 imgae가 corrupt 된다고 이해하면 됨.
<br>
<br>

Forward process simulation 예시.
>
$X \in \mathbb{R}^2, f=[1,0]^T, g=I$ 인 경우 시간에 따른 $X_t$를 plot함
![](https://velog.velcdn.com/images/syleekr/post/4c36071e-8557-4b43-92b0-40c0114efa43/image.png)
![](https://velog.velcdn.com/images/syleekr/post/419750bf-a31f-46a9-b11f-ed7f66f2d298/image.png)
![](https://velog.velcdn.com/images/syleekr/post/2dfaeb6a-b5f2-4676-8a76-9409bda9ef4d/image.png)
![](https://velog.velcdn.com/images/syleekr/post/bbf037d6-b855-42bb-b084-43b623d1035f/image.png)
- $f=[1,0]^T$, 즉 x축으로 1, y축으로 0인 상황이므로 시간이 지남에 따라 $X_t$는 전반적으로 +x축으로 이동하는 경향을 보이고 있다.
- $Z_k$ 항에 의하여 매 timestep마다 randomness가 더해지고 있다. 따라서 시간이 지남에 따라 점들이 더 확산되고 있다.
- **"Diffusion"**

### 2-2. Reverse process

Forward process를 정의했으니, 이 forward process(SDE)를 reverse 할 수 있는 방법만 알 수 있다면 끝.

가장 간단하게 생각해볼 수 있는 것은, ODE를 reverse 할때처럼 부호를 바꿔서 iterate하는 것. 즉
>
$X_{k+1} = X_{k} - \Delta{t}f(X_k , k \Delta t) - g(k \Delta t)\sqrt{\Delta t}Z_k$

그러나, 이는 SDE를 reverse 하지 못함. 위의 example에 대해서 위 식을 적용해보면,
![](https://velog.velcdn.com/images/syleekr/post/aaf13615-c882-4f6e-a6ff-3c434e58f2ce/image.png)
$- \Delta{t}f(X_k , k \Delta t)$ 항에 의하여 점들이 전반적으로 -x축으로 이동은 함. 그러나 $- g(k \Delta t)\sqrt{\Delta t}Z_k$ 항은 점들이 다시 cluster로 모이는 것을 장려하기는 커녕, $Z_k$항에 의해 오히려 더욱더 확산,diffuse된다...
<br>

올바른 방법은 다음 reverse-time SDE를 고려하는 것이다.
(by Anderson's reverse-time SDE theorem & Fokker-Plank equation )
> 
$dX_t = (f(X_t, t)-g^2(t) \nabla_{x}logp_t(X_t))dt + g(t)dW_t$
- $p_t:$ marginal probability distribution of $X_t$

즉 다음과 같이 compute 한다는 말임

>
$X_{k-1} = X_{k} - \Delta{t}(f(X_k, k \Delta t)-g^2(k \Delta t) \nabla_{x}logp_{k\Delta{t}}(X_k)) + g(k \Delta t)\sqrt{\Delta t}Z_k$

즉 reverse process는, $X_T, \ f, \ g, \ Z_T$ 을 바탕으로 $X_{T-1}$을 계산하고,  $X_{T-1}, \ f, \ g, \ Z_{T-1}$ 을 바탕으로 $X_{T-2}$를 계산하고, ... , 이 과정을 반복하여 $X_T , X_{T-1}, ..., X_0$를 얻는것.
$X_T$를 noise 라고 생각하고 reverse process를 통해 image $X_0$ 를 generate한다고 생각하면 된다.
위의 방법으로 reverse process를 진행하면 예시의 점들이 한 cluster로 다시 모이는 것을 확인할 수 있다.
![](https://velog.velcdn.com/images/syleekr/post/e4b02095-113c-4b71-93d8-cbaa3df8d9bc/image.png)

>
**이제 사실상 Diffusion Model의 high level view에 대한 설명은 끝난것.
$f,g,T$를 정해 forward process를 정의하고, $X_T \sim p_T$로 $X_T$를 sampling 한 후 reverse process에 따라 $X_0$를 계산하면, 새로운 데이터를 generate하는 것! **

### 2-3. VE and VP SDEs
Forward process로 보통 VE/VP SDE가 사용됨.
VE(variance exploding) SDE는 다음과 같이 주어짐
>
$dX_t = \sigma dW_t$
- 즉 $f=0, g : scalar$ 인 놈들
- 계산해보면, $X_t \mid X_0  \sim N(X_0, t\sigma^2I)$

VP(variance preserving) SDE는 다음과 같이 주어짐
>
$dX_t = -\beta X_tdt+\sigma dW_t$
- 즉 $f \neq 0, g : scalar$ 인 놈들
- 계산해보면, $X_t \mid X_0  \sim N(e^{-\beta t}X_0, {\sigma^2 \over {2\beta}}(1-e^{-2\beta t})I \ )$

>
즉 VE, VP SDE는 다음 성질을 가지는 다룰만한 SDE들임
$X_t  \overset{D}=\gamma_tX_0 - \sigma_t \varepsilon, \quad \varepsilon \sim N(0,I)$

## 3. Diffusion via SDE
이제 diffusion model의 실질적인 training/sampling(image generation) 과정을 살피자.
>
$dX_t = (f-g^2 \nabla_{x}logp_t)dt + gdW_t$,$\quad X_T \sim p_T$

위 방법대로 $X_T$를 sampling 한후 reverse time SDE를 simulate하면 그만인데, 두가지 문제가 존재한다.
>
- $X_T$를 $p_T$로부터 sampling 해야 하므로 $p_T$를 알고 있어야 하지만 그렇지 못한 상태.
- $\nabla_{x}logp_t$를  계산할 수 있어야 하는데, 모름.

따라서, 다음과 같이 해결한다.
>
- $f,g,T$를 잘 정하면 $p_T \sim N(0,\sigma^2_TI)$ ($\sigma_T$ : known)가 되도록 할 수 있다.
- $\nabla_{x}logp_t$를 모르니깐 이 값을 neural network $s_{\theta}(x,t)$로 근사하자.

즉 Diffusion model에서 딥러닝 = score network $s_{\theta}(x,t)$ 학습시키기.
score network 학습만 어떻게 하는지 알면 diffusion model 끝.

### 3-1. Training
Evaluate 할 수 있는, suitable loss function 만 있으면 DL training은 끝(autodiff가 알아서 함).
직관적으로 다음 loss를 생각해볼 수 있다.
>
$L(\theta)=\int_0^T{\lambda(t)\mathbb{E_{X_T}[\left\| s_{\theta}(X_t,t)- \nabla_{X_T}logp_t(X_T)\right\|^2]}}dt$ 
- $\lambda(t)$ : just some weighing factor

모든 시간(0~T)에서 score network 와 $\nabla_{x}logp_t$의 차이를 줄이고자 하는 매우 직관적인 loss function이다. 그런데 이 loss function은 evaluate 하는 것이 불가능하다. $\nabla_{X_T}logp_t(X_T)$를 모르니깐 이를 근사하도록 score network 를 train 시키고 싶어서 이 짓을 하는건데, loss function 자체에 우리가 모르는 $\nabla_{X_T}logp_t(X_T)$가 들어있다...

그렇기에 우리는 다음의 equivalent loss를 사용한다.
>
- $L(\theta)=\int_0^T{\lambda(t)\mathbb{E_{X_0}}\mathbb{E_{X_T \mid X_0}[\left\| s_{\theta}(X_t,t)- \nabla_{X_T}logp_{t \mid 0}(X_T \mid X_0)\right\|^2]}}dt$ 

원래 loss function과 매우 비슷해 보이지만, 우리가 모르는 항인 $\nabla_{X_T}logp_t(X_T)$가 $\nabla_{X_T}logp_{t \mid 0}(X_T \mid X_0)$ 로 변한 것을 볼 수 있다. 둘다 모르는거 아니냐? 라고 생각할 수도 있지만, $p_{t \mid 0}$는 $p_t$와 다르게 $f,g$를 잘 정하면 계산이 가능하다. 그 예시가 바로 VE,VP SDE!
얘네를 떠올려보면 $X_t  \overset{D}=\gamma_tX_0 - \sigma_t \varepsilon$ 의 관계를 만족하고 있었다. 다시 말해 $X_t \mid X_0$ 가 정규분포(multivariate gaussian)를 따르고 있다는 것인데, 우리는 정규분포 식을 잘 안다. 즉 $\nabla_{X_T}logp_{t \mid 0}(X_T \mid X_0)$ 는 계산 가능한 놈이고, 실제로 해보면 다음과 같이 주어진다.
>
$\nabla_{X_T}logp_{t \mid 0}(X_T \mid X_0)={\gamma_tX_0-X_t \over \sigma_t^2}\overset{D}={\varepsilon \over \sigma_t}$

이제 대입해보면, 우리의 loss function은 
>
- Define $\varepsilon_{\theta}(X_t,t) = \sigma_t s_{\theta}(X_t,t)$
- $L(\theta)=\int_0^T{\lambda(t)\mathbb{E_{X_0}}\mathbb{E_{X_T \mid X_0}[\left\| s_{\theta}(X_t,t)- \nabla_{X_T}logp_{t \mid 0}(X_T \mid X_0)\right\|^2]}}dt \\
\quad \quad = \int_0^T{{\lambda(t)\over \sigma_t^2} \mathbb{E_{X_0}}\mathbb{E}_{\varepsilon \sim N(0,I)}[\left\| \varepsilon_{\theta}(\gamma_tX_0-\sigma_t\varepsilon,t)- \varepsilon\right\|^2]}dt \\
\quad \quad = T\mathbb{E}_{t \sim Uniform[0,T]}\mathbb{E}_{X_0 \sim p_0}\mathbb{E}_{\varepsilon \sim N(0,I)}[{\lambda(t)\over \sigma_t^2}\left\| \varepsilon_{\theta}(\gamma_tX_0-\sigma_t\varepsilon,t)- \varepsilon\right\|^2 ]$ 

이제 training 은 다음과 같이 하면 끝!

>
$While\ not\ converged:$
$\quad X_0 \sim p_0$
$\quad t \sim Uniform[0,T]$
$\quad \varepsilon \sim N(0,I)$  
$\quad X_t = \gamma_tX_0 - \sigma_t \varepsilon$
$\quad Use \ optimizer\ to\ calculate\ and\ update\ respect\ to\ {\lambda(t)\over \sigma_t^2}\nabla_{\theta}\left\| \varepsilon_{\theta}(X_t,t)- \varepsilon\right\|^2$

즉 위와 같이 train을 해주면 epsilon network 를 얻게 되고, 이 epsilon network를 이용하여 $\nabla_{x}logp_t$를 근사하는 score network를 얻을 수 있다.

### 3-3.Sampling

이제 diffusion model을 이용하여 image(data) generation을 하자. 2-2의 식을 계산하기 위해 train 이 완료된 score network를 사용하면 그만이다.
>
$X_K \sim N(0, \sigma_T^2I)$
$for\ k = K,K-1,...,2,1:$
$\quad Z_k \sim N(0,I)$
$\quad X_{k-1} = X_{k} - \Delta{t}(-\beta X_k-\sigma^2 s_{\theta}(X_k,k\Delta t)) + \sigma\sqrt{\Delta t}Z_k$ 
- $X_0 : generated\ image!$


**Diffusion model 끝!**
