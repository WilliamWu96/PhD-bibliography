# PhD-bibliography

# Table of contents

* [Reinforcement Learning](#reinforcement-learning)
  * [Theory](#theory) 
  * [Value based](#value-based)
  * [Policy Based](#policy-based)
  
* [Offline Reinforcement Learning](#offline-reinforcement-learning)

* [Continual Reinforcement Learning](#continual-reinforcement-learning)

* [Machine Learning](#machine-learning)
  * [Theory](#theory) 

* [RL Environment Setting](#rl-environment-setting)

* [Generalization](#generalization-dart)
  * [RL Generalization](#rl-generalization)
  * [Data Augmentation (state-based)](#data-augmentation-state-based)
  * [Representation Learning](#representation-learning)
  * [IL Generalization](#il-generalization)

* [Domain Adaptation (DA)](#domain-adaptation-da)
  * [Metric learning-based DA](#metric-learning-based-da)
  * [Adversarial learning-based DA](#adversarial-learning-based-da)
  * [Domain adaptation in RL](#Domain-adaptation-in-RL)
    * [Domain randomization](#domain-randomization)
    * [GAN-based DA](#GAN-based-DA)  
    * [VAE-based DA](#VAE-based-DA)

* [Transfer Learning](#transfer-learning)
  
* [Robustness](#robustness)
  * [Adversarial Robustness](#Adversarial-Robustness)
  * [RL Robustness](#RL-Robustness)

* [Uncertainty](#uncertainty)


* [Bayesian Optimization](#bayesian-optimization)
  * [BO Tutorial](#bo-tutorial)


* [Gaussian Processes](#gaussian-processes)
  * [GP Book](#gp-book)
  * [GP Tutorial](#gp-tutorial)
  * [Deep Gaussian Processes](#deep-gaussian-processes)
 
* [Exploration Strategy](#exploration-strategy)
  * [Count-based Exploration Strategy](#Count-based-exploration-strategy)
  * [Prediction-based Exploration Strategy](#Prediction-based-Exploration-Strategy)
  * [Unsupervised Active Pretraining](#Unsupervised-Active-Pretraining)
* [CARLA](#carla)
  * [Reinforcement Learning CARLA](#reinforcement-learning-carla)
  * [Imitation Learning CARLA](#imitation-learning-carla)
  
* [RL Safety](#RL-SAFETY)
* [Robots Appilcation](#robots-application)
  * [Github Demo](#github-demo)

# Reinforcement Learning

## Theory
* [Reinforcement learning: A survey](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
* [A Markovian Decision Process](https://www.jstor.org/stable/24900506?seq=1)
* [Planning and acting in partially observable stochastic domains](https://www.sciencedirect.com/science/article/pii/S000437029800023X)

## Classic Papers
* [DQN] [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [PPO 2017] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
* [SAC 2018] [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
* [SAC 2018] [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
* [DDPG] [CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING](https://arxiv.org/pdf/1509.02971.pdf)

# Offline Reinforcement Learning
* [D4RL: DATASETS FOR DEEP DATA-DRIVEN REINFORCEMENT LEARNING](https://arxiv.org/pdf/2004.07219.pdf)
* [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/pdf/2006.04779.pdf)
* [Off-Policy Deep Reinforcement Learning without Exploration](https://arxiv.org/pdf/1812.02900.pdf)
* [General Batch RL papers](https://github.com/apexrl/Batch-Offline--RL-Paper-Lists#general-batch-rl )

# Continual Reinforcement Learning
* [Continual World: A Robotic Benchmark For Continual Reinforcement Learning](https://arxiv.org/pdf/2105.10919.pdf) [website](https://sites.google.com/view/continualworld)
* [Continual Adaptation of Visual Representations via Domain Randomization and Meta-learning](https://arxiv.org/pdf/2012.04324.pdf)
* [PackNet] [PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://arxiv.org/pdf/1711.05769.pdf)
* [VCL] [VARIATIONAL CONTINUAL LEARNING](https://arxiv.org/pdf/1710.10628.pdf)
* [L2/EWC] [Overcoming catastrophic forgetting inneural networks](https://www.pnas.org/doi/epdf/10.1073/pnas.1611835114)
* [MAS] [Memory Aware Synapses: Learning what (not) to forget](https://arxiv.org/pdf/1711.09601.pdf)
* [A-GEM] [EFFICIENT LIFELONG LEARNING WITH A-GEM](https://arxiv.org/pdf/1812.00420.pdf)
* [PopArt] [Multi-task deep reinforcement learning with popart](https://arxiv.org/pdf/1809.04474.pdf)
* [Task-Agnostic Continual Reinforcement Learning: In Praise of a Simple Baseline](https://arxiv.org/pdf/2205.14495.pdf)
* [Awesome Incremental Learning / Lifelong learning](https://github.com/xialeiliu/Awesome-Incremental-Learning)
* [CoLLAs] [Conference on Lifelong Learning Agents](https://virtual.lifelong-ml.cc/papers.html?session=Conference&filter=keywords)

# Machine Learning

## Theory
* [GAN] [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
* [Transformer] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

# RL Environment Setting
* [gym] (https://github.com/openai/gym)
* [highway-env] (https://github.com/eleurent/highway-env)
* [Procgen Benchmark] (https://github.com/openai/procgen)
* [DI-drive] (https://github.com/opendilab/DI-drive)
* [D4RL,offline RL benchmark] (https://arxiv.org/pdf/2004.07219.pdf)

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
* [A Regularized Approach to Sparse Optimal Policy in Reinforcement Learning](https://arxiv.org/abs/1903.00725)
* [Improving Generalization in Reinforcement Learning with Mixture Regularization](https://proceedings.neurips.cc/paper/2020/file/5a751d6a0b6ef05cfe51b86e5d1458e6-Paper.pdf) [[site]](https://policy.fit/projects/mixreg/index.html) [[code]](https://github.com/kaixin96/mixreg)

## Data Augmentation (state-based)
* [mixup] [mixup: BEYOND EMPIRICAL RISK MINIMIZATION](https://arxiv.org/pdf/1710.09412.pdf)
* [mixreg] [Improving Generalization in Reinforcement Learning with Mixture Regularization](https://arxiv.org/pdf/2010.10814v1.pdf)
* [NMER] [Neighborhood Mixup Experience Replay: Local Convex Interpolation for Improved Sample Efficiency in Continuous Control Tasks](https://proceedings.mlr.press/v168/sander22a/sander22a.pdf)
* [Augmix] [AUGMIX: A SIMPLE DATA PROCESSING METHOD TO IMPROVE ROBUSTNESS AND UNCERTAINTY](https://arxiv.org/pdf/1912.02781.pdf)
* [Augmax] [AugMax: Adversarial Composition of Random Augmentations for Robust Training](https://proceedings.neurips.cc/paper/2021/file/01e9565cecc4e989123f9620c1d09c09-Paper.pdf)
* [Continuous Transition] [Continuous Transition: Improving Sample Efficiency for Continuous Control Problems via MixUp](https://arxiv.org/pdf/2011.14487.pdf)
* [S4RL] [S4RL: Surprisingly Simple Self-Supervision for Offline Reinforcement Learning in Robotics](https://arxiv.org/pdf/2103.06326.pdf)

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
* [Generalization in Reinforcement Learning with Selective Noise Injection and Information Bottleneck](https://arxiv.org/abs/1910.12911)

## IL generalization

* [arXiv 2021] [Generalization Guarantees for Imitation Learning](https://arxiv.org/pdf/2008.01913)

# Domain Generalization (DG)

* [Domain Generalization: A Survey](https://arxiv.org/abs/2103.02503)

# Domain Adaptation (DA)

* [A Comprehensive Survey on Transfer Learning](https://arxiv.org/pdf/1911.02685.pdf)
* [A Review of Single-Source Deep Unsupervised Visual Domain Adaptation](https://arxiv.org/pdf/2009.00155.pdf)
* [Active Domain Randomization](http://proceedings.mlr.press/v100/mehta20a/mehta20a.pdf) [[code]](https://github.com/montrealrobotics/active-domainrand)

## Continual Domain Adaptation
* [Continual Adaptation of Visual Representations via Domain Randomization and Meta-learning](https://arxiv.org/pdf/2012.04324.pdf) [website](https://europe.naverlabs.com/research/computer-vision/continual-adaptation-of-visual-representations-via-domain-randomization-and-meta-learning/)
* [Same State, Different Task: Continual Reinforcement Learning without Interference](https://arxiv.org/pdf/2106.02940.pdf)
* [La-MAML: Look-ahead Meta Learning for Continual Learning](https://proceedings.neurips.cc/paper/2020/file/85b9a5ac91cd629bd3afe396ec07270a-Paper.pdf)
* [Meta-Learning Representations for Continual Learning](https://papers.nips.cc/paper/2019/file/f4dd765c12f2ef67f98f3558c282a9cd-Paper.pdf)

## Metric learning-based DA

* [Central Moment Discrepancy (CMD) for Domain-Invariant Representation Learning](https://arxiv.org/abs/1702.08811)
* [Integrating structured biological data by kernel maximum mean discrepancy](https://academic.oup.com/bioinformatics/article/22/14/e49/228383)

## Adversarial learning-based DA

* [Divergence-agnostic Unsupervised Domain daptation by Adversarial Attacks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9528987)
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
* [Active Domain Randomization](https://arxiv.org/pdf/1904.04762.pdf)

### GAN-based DA
* [Unsupervised Image-to-Image Translation Networks](https://arxiv.org/abs/1703.00848)
* [Unpaired image-to-image translation using cycle-consistent adversarial networks](https://arxiv.org/abs/1703.10593)
* [Virtual to real reinforcement learning for autonomous driving.](https://arxiv.org/abs/1704.03952)
* [Adapting deep visuomotor representations with weak pairwise constraints](https://arxiv.org/abs/1511.07111)
* [Transfer Learning for Related Reinforcement Learning Tasks via Image-to-Image Translation](https://arxiv.org/abs/1806.07377) [[code]](https://github.com/ShaniGam/RL-GAN)


### VAE-based DA
* [DARLA: Improving Zero-Shot Transfer in Reinforcement Learning](https://arxiv.org/abs/1707.08475)

# Transfer Learning
* [Awesome Transfer Learning Papers](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#transfer-reinforcement-learning)

# Robustness

## Adversarial Robustness
* [DataAug+AdversarialTraining] [Fixing Data Augmentation to Improve Adversarial Robustness](https://arxiv.org/abs/2103.01946)
* [DataAug+AdversarialTraining] [Robustness and Accuracy Could Be Reconcilable by (Proper) Definition](https://arxiv.org/abs/2202.10103)
## RL Robustness
* [Maximum Entropy RL (Provably) Solves Some Robust RL Problems](https://openreview.net/pdf?id=PtSAD3caaA2)

# Uncertainty

* [A Survey of Uncertainty in Deep Neural Networks](https://arxiv.org/abs/2107.03342)

# Bayesian Optimization

## BO Tutorial
* [How to Implement Bayesian Optimization from Scratch in Python](https://machinelearningmastery.com/what-is-bayesian-optimization/)
* [The intuitions behind Bayesian Optimization with Gaussian Processes](https://towardsdatascience.com/the-intuitions-behind-bayesian-optimization-with-gaussian-processes-7e00fcc898a0)
* [Deeply Understanding on Bayesian Optimization - Chinese version](https://zhuanlan.zhihu.com/p/53826787)


# Gaussian Processes

* [Last Layer Marginal Likelihood for Invariance Learning](https://arxiv.org/abs/2106.07512)
* [Learning Invariant Weights in Neural Networks](https://arxiv.org/abs/2202.12439)

## GP Book
* [Gaussian Processes for Machine Learning](http://gaussianprocess.org/gpml/chapters/)

## GP Tutorial

* [Gaussian Processes are Not So Fancy](https://planspace.org/20181226-gaussian_processes_are_not_so_fancy/)
* [Gaussian Processes for Dummies](https://katbailey.github.io/post/gaussian-processes-for-dummies/)
* [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/)
* [Gaussian Processes in 2-dimensional model - jupyter notebook](https://nbviewer.org/github/SheffieldML/notebook/blob/master/GPy/basic_gp.ipynb)
* [The Kernel Cookbook: Advice on Covariance functions](https://www.cs.toronto.edu/~duvenaud/cookbook/)
* [Gaussian Processes for Classification With Python](https://machinelearningmastery.com/gaussian-processes-for-classification-with-python/)
* [An Intuitive Tutorial to Gaussian Processes Regression](https://arxiv.org/pdf/2009.10862.pdf)

## Deep Gaussian Processes

* [Deep Gaussian Processes](http://proceedings.mlr.press/v31/damianou13a.pdf)
* [Deep Gaussian Processes_Pytorch](https://docs.gpytorch.ai/en/v1.5.1/examples/05_Deep_Gaussian_Processes/Deep_Gaussian_Processes.html)
* [Deep convolutional Gaussian processes](https://arxiv.org/abs/1810.03052)

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

# Robots Application
* [Bayesian Meta-Learning for Few-Shot Policy Adaptation Across Robotic Platforms](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9636628)
* [Data-Efficient Reinforcement and Transfer Learning in Robotics](https://www.diva-portal.org/smash/get/diva2:1501390/FULLTEXT01.pdf)
* [Sim-to-Real Transfer with Incremental Environment Complexity for Reinforcement Learning of Depth-Based Robot Navigation](https://arxiv.org/pdf/2004.14684.pdf)
* [Reinforced Imitation: Sample Efficient Deep Reinforcement Learning for Map-less Navigation by Leveraging Prior Demonstrations](https://arxiv.org/pdf/1805.07095.pdf)
* [Target-Driven Mapless Navigation for Self-Driving Car](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9641134)

## Github Demo
* [Reinforced Imitation: Sample Efficient Deep Reinforcement Learning for Map-less Navigation by Leveraging Prior Demonstrations](https://github.com/ethz-asl/rl-navigation)
* [RL Navigation for Mantis and Turtlebot3](https://github.com/bhctsntrk/mantis_ddqn_navigation)
* [Mobile Robot Planner with Low-cost Cameras Using Deep Reinforcement Learning](https://github.com/trqminh/rl-mapless-navigation)
