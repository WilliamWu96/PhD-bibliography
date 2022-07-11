# PhD-bibliography

# Table of contents

* [Reinforcement Learning](#reinforcement-learning)
  * [Theory](#theory) 
  * [Value based](#value-based)
  * [Policy Based](#policy-based)

* [Machine Learning](#machine-learning)
  * [Theory](#theory) 


* [Generalization](#generalization-dart)
  * [RL Generalization](#rl-generalization)
  * [Representation Learning](#representation-learning)
  * [IL Generalization](#il-generalization)

* [Domain Adaptation (DA)](#domain-adaptation-da)
  * [Metric learning-based DA](#metric-learning-based-da)
  * [Adversarial learning-based DA](#adversarial-learning-based-da)
  * [Domain adaptation in RL](#Domain-adaptation-in-RL)
    * [Domain randomization](#domain-randomization)
    * [GAN-based DA](#GAN-based-DA)  
    * [VAE-based DA](#VAE-based-DA)

* [Gaussian Processes](#gaussian-processes)
  * [GP Book](#gp-book)
  * [GP Tutorial](#gp-tutorial)
  * [Deep Gaussian Processes](#deep-gaussian-processes)
  
* [Robustness](#robustness)
  * [Adversarial Robustness](#Adversarial-Robustness)
  * [RL Robustness](#RL-Robustness)

* [Uncertainty](#uncertainty)


* [Gaussian Process](#gaussian-process)
* [Exploration Strategy](#exploration-strategy)
  * [Count-based Exploration Strategy](#Count-based-exploration-strategy)
  * [Prediction-based Exploration Strategy](#Prediction-based-Exploration-Strategy)
  * [Unsupervised Active Pretraining](#Unsupervised-Active-Pretraining)
* [CARLA](#carla)
  * [Reinforcement Learning CARLA](#reinforcement-learning-carla)
  * [Imitation Learning CARLA](#imitation-learning-carla)
* [RL Safety](#RL-SAFETY)

# Reinforcement Learning

## Theory
* [Reinforcement learning: A survey](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
* [A Markovian Decision Process](https://www.jstor.org/stable/24900506?seq=1)
* [Planning and acting in partially observable stochastic domains](https://www.sciencedirect.com/science/article/pii/S000437029800023X)

## Value based
* [DQN] [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

## Policy based
### Policy gradient
* [PPO 2017] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
* [SAC 2018] [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
* [SAC 2018] [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)

# Machine Learning

## Theory
* [GAN] [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
* [Transformer] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

# Generalization :dart:

## RL generalization

* [arXiv 2021] [A Survey of Generalisation in Deep Reinforcement Learning](https://arxiv.org/abs/2111.09794)
* [arXiv 2021] [Generalization of Reinforcement Learning with Policy-Aware Adversarial Data Augmentation](https://arxiv.org/abs/2106.15587)
  <details>
    <summary> Hanping Zhang et al. 
        <em></em> 2021 </summary>
    The generalization gap in reinforcement learning (RL) has been a significant obstacle that prevents the RL agent from learning general skills and adapting to varying environments. Increasing the generalization capacity of the RL systems can significantly improve their performance on real-world working environments. In this work, we propose a novel policy-aware adversarial data augmentation method to augment the standard policy learning method with automatically generated trajectory data. Different from the commonly used observation transformation based data augmentations, our proposed method adversarially generates new trajectory data based on the policy gradient objective and aims to more effectively increase the RL agent's generalization ability with the policy-aware data augmentation. Moreover, we further deploy a mixup step to integrate the original and generated data to enhance the generalization capacity while mitigating the over-deviation of the adversarial data. We conduct experiments on a number of RL tasks to investigate the generalization performance of the proposed method by comparing it with the standard baselines and the state-of-the-art mixreg approach. The results show our method can generalize well with limited training diversity, and achieve the state-of-the-art generalization test performance.
    </details>
    
* [arXiv 2021] [Learning Invariant Representations for Reinforcement Learning without Reconstruction](https://arxiv.org/abs/2006.10742)
  <details>
    <summary> Amy Zhang et al. 
        <em>ICLR</em> 2021 </summary>
    We study how representation learning can accelerate reinforcement learning from rich observations, such as images, without relying either on domain knowledge or pixel-reconstruction. Our goal is to learn representations that both provide for effective downstream control and invariance to task-irrelevant details. Bisimulation metrics quantify behavioral similarity between states in continuous MDPs, which we propose using to learn robust latent representations which encode only the task-relevant information from observations. Our method trains encoders such that distances in latent space equal bisimulation distances in state space. We demonstrate the effectiveness of our method at disregarding task-irrelevant information using modified visual MuJoCo tasks, where the background is replaced with moving distractors and natural videos, while achieving SOTA performance. We also test a first-person highway driving task where our method learns invariance to clouds, weather, and time of day. Finally, we provide generalization results drawn from properties of bisimulation metrics, and links to causal inference.
    </details>
    
* [RAD 2020] [Reinforcement Learning with Augmented Data](https://arxiv.org/abs/2004.14990)
    
* [DrQ 2020] [Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels](https://arxiv.org/abs/2004.13649), Ilya Kostrikov 
* [DrQv2 2021] [Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning](https://arxiv.org/abs/2107.09645)
* [SVEA] [Stabilizing Deep Q-Learning with ConvNets and Vision Transformers under Data Augmentation](https://arxiv.org/abs/2107.00644)
* [On the Generalization Gap in Reparameterizable Reinforcement Learning](https://arxiv.org/abs/1905.12654)
## Representation Learning
* [CURL] [CURL: Contrastive Unsupervised Representations for Reinforcement Learning](https://arxiv.org/abs/2004.04136)
* [BayLime] [BayLIME: Bayesian Local Interpretable Model-Agnostic Explanations](https://arxiv.org/abs/2012.03058)
* [arXiv 2021] [Learning Invariant Representations for Reinforcement Learning without Reconstruction](https://arxiv.org/abs/2006.10742)
  <details>
    <summary> Amy Zhang et al. 
        <em>ICLR</em> 2021 </summary>
    We study how representation learning can accelerate reinforcement learning from rich observations, such as images, without relying either on domain knowledge or pixel-reconstruction. Our goal is to learn representations that both provide for effective downstream control and invariance to task-irrelevant details. Bisimulation metrics quantify behavioral similarity between states in continuous MDPs, which we propose using to learn robust latent representations which encode only the task-relevant information from observations. Our method trains encoders such that distances in latent space equal bisimulation distances in state space. We demonstrate the effectiveness of our method at disregarding task-irrelevant information using modified visual MuJoCo tasks, where the background is replaced with moving distractors and natural videos, while achieving SOTA performance. We also test a first-person highway driving task where our method learns invariance to clouds, weather, and time of day. Finally, we provide generalization results drawn from properties of bisimulation metrics, and links to causal inference.
    </details>
* [2021] [Contrastive Behavioral Similarity Embeddings for Generalization in Reinforcement Learning](https://arxiv.org/abs/2101.05265)

## IL generalization

* [arXiv 2021] [Generalization Guarantees for Imitation Learning](https://arxiv.org/pdf/2008.01913)

# Domain Adaptation (DA)

* [Divergence-agnostic Unsupervised Domain daptation by Adversarial Attacks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9528987)
* [A Comprehensive Survey on Transfer Learning](https://arxiv.org/pdf/1911.02685.pdf)
* [A Review of Single-Source Deep Unsupervised Visual Domain Adaptation](https://arxiv.org/pdf/2009.00155.pdf)

## Metric learning-based DA

* [Central Moment Discrepancy (CMD) for Domain-Invariant Representation Learning](https://arxiv.org/abs/1702.08811)
* [Integrating structured biological data by kernel maximum mean discrepancy](https://academic.oup.com/bioinformatics/article/22/14/e49/228383)

## Adversarial learning-based DA
* [Unsupervised domain adaptation by backpropagation](https://arxiv.org/abs/1409.7495)
* [Conditional Adversarial Domain Adaptation](https://arxiv.org/abs/1705.10667)

## Domain adaptation in RL
* [Domain Adaptation In Reinforcement Learning Via Latent Unified State Representation](https://arxiv.org/pdf/2102.05714.pdf)
* [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)
* [awesome-domain-adaptation-by-zhaoxin](https://github.com/zhaoxin94/awesome-domain-adaptation)

### Domain randomization
* [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907)
* [Learning Dexterous In-Hand Manipulation](https://arxiv.org/abs/1808.00177)
* [Robust Visual Domain Randomization for Reinforcement Learning](https://arxiv.org/abs/1910.10537)

### GAN-based DA
* [Unsupervised Image-to-Image Translation Networks](https://arxiv.org/abs/1703.00848)
* [Unpaired image-to-image translation using cycle-consistent adversarial networks](https://arxiv.org/abs/1703.10593)
* [Virtual to real reinforcement learning for autonomous driving.](https://arxiv.org/abs/1704.03952)
* [Adapting deep visuomotor representations with weak pairwise constraints](https://arxiv.org/abs/1511.07111)
* [Transfer Learning for Related Reinforcement Learning Tasks via Image-to-Image Translation](https://arxiv.org/abs/1806.07377)


### VAE-based DA
* [DARLA: Improving Zero-Shot Transfer in Reinforcement Learning](https://arxiv.org/abs/1707.08475)

# Gaussian Processes

## GP Book
* [Gaussian Processes for Machine Learning](http://gaussianprocess.org/gpml/chapters/)

## GP Tutorial

* [Gaussian Processes are Not So Fancy](https://planspace.org/20181226-gaussian_processes_are_not_so_fancy/)
* [Gaussian Processes for Dummies](https://katbailey.github.io/post/gaussian-processes-for-dummies/)
* [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/)
* [Gaussian Processes in 2-dimensional model - jupyter notebook](https://nbviewer.org/github/SheffieldML/notebook/blob/master/GPy/basic_gp.ipynb)
* [The Kernel Cookbook: Advice on Covariance functions](https://www.cs.toronto.edu/~duvenaud/cookbook/)

## Deep Gaussian Processes

* [Deep Gaussian Processes](http://proceedings.mlr.press/v31/damianou13a.pdf)
* [Deep Gaussian Processes_Pytorch](https://docs.gpytorch.ai/en/v1.5.1/examples/05_Deep_Gaussian_Processes/Deep_Gaussian_Processes.html)
* [Deep convolutional Gaussian processes](https://arxiv.org/abs/1810.03052)

# Robustness

## Adversarial Robustness
* [DataAug+AdversarialTraining] [Fixing Data Augmentation to Improve Adversarial Robustness](https://arxiv.org/abs/2103.01946)
* [DataAug+AdversarialTraining] [Robustness and Accuracy Could Be Reconcilable by (Proper) Definition](https://arxiv.org/abs/2202.10103)
## RL Robustness
* [Maximum Entropy RL (Provably) Solves Some Robust RL Problems](https://openreview.net/pdf?id=PtSAD3caaA2)

# Uncertainty

* [A Survey of Uncertainty in Deep Neural Networks](https://arxiv.org/abs/2107.03342)

# Gaussian Process

* [Last Layer Marginal Likelihood for Invariance Learning](https://arxiv.org/abs/2106.07512)
* [Learning Invariant Weights in Neural Networks](https://arxiv.org/abs/2202.12439)


# Exploration Strategy

A list of papers regarding exploration strategy in (deep) reinforcement learning.

## Count-based Exploration Strategy
* [Unifying count-based exploration and intrinsic motivation](https://arxiv.org/abs/1606.01868)
* [Count-Based Exploration with Neural Density Models](https://arxiv.org/abs/1703.01310)
* [#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning](https://arxiv.org/abs/1611.04717)
* [Contingency-aware exploration in reinforcement learning](https://arxiv.org/abs/1811.01483)

## Prediction-based Exploration Strategy

* [Exploration Strategies in Deep Reinforcement Learning](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/), Lil, (2020)
* [RND] [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)
* [Episodic Curiosity] [Episodic Curiosity through Reachability](https://arxiv.org/abs/1810.02274), (2018)


## Unsupervised Active Pretraining

* [APT] [Behavior From the Void: Unsupervised Active Pre-Training](https://arxiv.org/abs/2103.04551)

* [APS] [APS: Active Pretraining with Successor Features](https://arxiv.org/abs/2108.13956)

* [arXiv 2021] [Reinforcement Learning with Prototypical Representations](https://arxiv.org/abs/2102.11271)

* [RL Active Learning] [Learning with not Enough Data Part 2: Active Learning](https://lilianweng.github.io/posts/2022-02-20-active-learning/)


# CARLA

## Reinforcement Learning CARLA

* [RAIL 2021] [Learning to drive from a world on rails](https://arxiv.org/pdf/2105.00636)

* [PhD thesis 2020] [Safe and Efficient Reinforcement Learning for Behavioural Planning in Autonomous Driving](https://hal.inria.fr/tel-03035705/document)

* [2019] [End-to-End Model-Free Reinforcement Learning for Urban Driving using Implicit Affordances](https://arxiv.org/abs/1911.10868)

## Imitation Learning CARLA

* [LAV 2022] [Learning from All Vehicles](https://arxiv.org/pdf/2203.11934)

* [cheating 2021] [Learning by cheating](https://arxiv.org/pdf/1912.12294)

* [TransFuser 2021] [Multi-Modal Fusion Transformer for End-to-End Autonomous Driving](https://arxiv.org/abs/2104.09224)

# RL Safety
* [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565)

