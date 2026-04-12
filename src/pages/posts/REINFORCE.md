---
layout: ../../layouts/PostLayout.astro
title: 公式推导：REINFORCE
description: 尝试推导出 REINFORCE 算法的参数更新公式
date: 2026-4-12
---

我想该博客中尝试推导REINFORCE的参数更新公式，巩固一下自己对REINFORCE的理解。


我们首先设定损失函数：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]
$$

这里：

- $\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_T)$ 表示一条完整轨迹
- $R(\tau)$ 表示轨迹的总回报
- $\pi_\theta(a \mid s)$ 表示参数为 $\theta$ 的策略函数

若采用折扣回报，则：

$$
R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_{t+1}
$$

由于轨迹是按策略随机采样得到的，因此：

$$
J(\theta) = \sum_{\tau} P(\tau; \theta)\, R(\tau)
$$

其中 $P(\tau; \theta)$ 表示在参数为 $\theta$ 的策略下生成轨迹 $\tau$ 的概率。

于是：

$$
\nabla_\theta J(\theta)
= \nabla_\theta \sum_{\tau} P(\tau; \theta) R(\tau)
= \sum_{\tau} \nabla_\theta P(\tau; \theta)\, R(\tau)
$$

利用恒等式：

$$
\nabla_\theta P(\tau; \theta)
= P(\tau; \theta)\, \nabla_\theta \log P(\tau; \theta)
$$

代入上式，得到：

$$
\nabla_\theta J(\theta)
= \sum_{\tau} P(\tau; \theta)\, \nabla_\theta \log P(\tau; \theta)\, R(\tau)
$$

写成期望形式：

$$
\nabla_\theta J(\theta)
= \mathbb{E}_{\tau \sim \pi_\theta}
\left[
R(\tau)\, \nabla_\theta \log P(\tau; \theta)
\right]
$$

---
**此处，我们可以对 $\nabla_\theta \log P(\tau; \theta)$ 进行分析。**
轨迹概率可以写成：

$$
P(\tau; \theta)
= \rho(s_0)\prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t)\, P(s_{t+1} \mid s_t, a_t)
$$

对数化：

$$
\log P(\tau; \theta)
= \log \rho(s_0)
+ \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t)
+ \sum_{t=0}^{T-1} \log P(s_{t+1} \mid s_t, a_t)
$$

对 $\theta$ 求梯度：

$$
\nabla_\theta \log P(\tau; \theta)
= \nabla_\theta \log \rho(s_0)
+ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)
+ \sum_{t=0}^{T-1} \nabla_\theta \log P(s_{t+1} \mid s_t, a_t)
$$

由于：

- 初始状态分布 $\rho(s_0)$ 通常不依赖 $\theta$
- 环境转移概率 $P(s_{t+1} \mid s_t, a_t)$ 也不依赖 $\theta$

因此：

$$
\nabla_\theta \log P(\tau; \theta)
= \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

代回梯度表达式：

$$
\nabla_\theta J(\theta)
= \mathbb{E}_{\tau \sim \pi_\theta}
\left[
R(\tau)\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)
\right]
$$

由此，我们得到了 **REINFORCE 的基本策略梯度形式**。

---

定义从时刻 $t$ 开始的回报：

$$
G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_{k+1}
$$

则可将梯度改写为：

$$
\nabla_\theta J(\theta)
= \mathbb{E}_{\tau \sim \pi_\theta}
\left[
\sum_{t=0}^{T-1}
G_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)
\right]
$$

**这个形式比直接使用整条轨迹总回报 $R(\tau)$ 更常见，因为它降低了方差，并且更符合“动作 $a_t$ 只影响其后的奖励”这一事实。**


实际训练时，通常使用采样得到的一条或多条轨迹来近似期望。若使用单条轨迹 $\tau$，则梯度估计为：

$$
\hat{\nabla}_\theta J(\theta)
=
\sum_{t=0}^{T-1}
G_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

由于我们要最大化 $J(\theta)$，因此使用梯度上升：

$$
\theta \leftarrow \theta + \alpha \hat{\nabla}_\theta J(\theta)
$$

代入上面的梯度估计式：

$$
\theta
\leftarrow
\theta
+
\alpha
\sum_{t=0}^{T-1}
G_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

**这就是最常见的 REINFORCE 参数更新公式。**

---

为了进一步降低方差，可以减去一个 baseline $b(s_t)$，得到：

$$
\theta
\leftarrow
\theta
+
\alpha
\sum_{t=0}^{T-1}
\left(G_t - b(s_t)\right)
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

当 baseline 取状态价值函数 $V^\pi(s_t)$ 时，可写为 advantage 形式：

$$
A_t = G_t - V^\pi(s_t)
$$

于是更新公式为：

$$
\theta
\leftarrow
\theta
+
\alpha
\sum_{t=0}^{T-1}
A_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$
