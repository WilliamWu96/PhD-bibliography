# PhD-bibliography

# Table of contents

* [Mathmatic Theory](#mathmatic-theory)
  
  * [Bayesian Optimization](#bayesian-optimization)
    * [BO Tutorial](#bo-tutorial)
    
  * [Gaussian Processes](#gaussian-processes)
    * [GP Book](#gp-book)
    * [GP Tutorial](#gp-tutorial)
    * [Deep Gaussian Processes](#deep-gaussian-processes)

  * [Uncertainty](#uncertainty)

* [Machine Learning](#machine-learning)
  * [Theory](#theory) 
  * [Blog](#blog)

* [Reinforcement Learning](#reinforcement-learning)
  * [Theory](#theory) 
  * [Classic Papers](#classic-paper)
  * [Trustworthy RL](#trustworthy-rl)
  * [RL Environment Setting](#rl-environment-setting)
  
* [Offline Reinforcement Learning](#offline-reinforcement-learning)

* [Continual Reinforcement Learning](#continual-reinforcement-learning)
  * [Bayesian Approaches: Continual RL](#Bayesian-Approaches-Continual-RL)
  * [Memory-based Continual learning](#memory-based-continual-learning)
  * [Gradient-based Continual Learning](#Gradient-based-Continual-Learning)

* [Generalization](#generalization-dart)
  * [ML Generalization](#ml-generalization)
  * [RL Generalization](#rl-generalization)
  * [Data Augmentation (state-based)](#data-augmentation-state-based)
  * [Model Perspective](#model-perspective)
  * [Representation Learning](#representation-learning)
    * [RL with GNN](#RL-with-GNN)
  * [IL Generalization](#il-generalization)

* [Domain Adaptation (DA)](#domain-adaptation-da)
  * [Metric learning-based DA](#metric-learning-based-da)
  * [Adversarial learning-based DA](#adversarial-learning-based-da)
  * [Domain adaptation in RL](#Domain-adaptation-in-RL)
    * [Domain randomization](#domain-randomization)
    * [GAN-based DA](#GAN-based-DA)  
    * [VAE-based DA](#VAE-based-DA)

* [Transfer Learning](#transfer-learning)

* [Causal Reasoning](#causal-reasoning)
  
* [Robustness](#robustness)
  * [Adversarial Attack](#Adversarial-Attack)
  * [Adversarial Robustness](#Adversarial-Robustness)
  * [RL Robustness](#RL-Robustness)
 
* [Exploration Strategy](#exploration-strategy)
  * [Count-based Exploration Strategy](#Count-based-exploration-strategy)
  * [Prediction-based Exploration Strategy](#Prediction-based-Exploration-Strategy)
  * [Unsupervised Active Pretraining](#Unsupervised-Active-Pretraining)

* [RL Application](#RL-Application)
  * [Traffic Problem based on RL](#traffic-problem-based-on-RL)
    * [Traffic env setting](#traffic-env-setting) 

  * [CARLA](#carla)
    * [Reinforcement Learning CARLA](#reinforcement-learning-carla)
    * [Imitation Learning CARLA](#imitation-learning-carla)
    * [3D/Nerf attack](#3D-Nerf-attack)

  * [RL Safety](#RL-SAFETY)
  * [Robots Appilcation](#robots-application)
    * [Github Demo](#github-demo)
# Mathmatic Theory

## Bayesian Optimization

### BO Tutorial
* [How to Implement Bayesian Optimization from Scratch in Python](https://machinelearningmastery.com/what-is-bayesian-optimization/)
* [The intuitions behind Bayesian Optimization with Gaussian Processes](https://towardsdatascience.com/the-intuitions-behind-bayesian-optimization-with-gaussian-processes-7e00fcc898a0)
* [Deeply Understanding on Bayesian Optimization - Chinese version](https://zhuanlan.zhihu.com/p/53826787)


## Gaussian Processes

* [Last Layer Marginal Likelihood for Invariance Learning](https://arxiv.org/abs/2106.07512)
* [Learning Invariant Weights in Neural Networks](https://arxiv.org/abs/2202.12439)

### GP Book
* [Gaussian Processes for Machine Learning](http://gaussianprocess.org/gpml/chapters/)

### GP Tutorial

* [Gaussian Processes are Not So Fancy](https://planspace.org/20181226-gaussian_processes_are_not_so_fancy/)
* [Gaussian Processes for Dummies](https://katbailey.github.io/post/gaussian-processes-for-dummies/)
* [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/)
* [Gaussian Processes in 2-dimensional model - jupyter notebook](https://nbviewer.org/github/SheffieldML/notebook/blob/master/GPy/basic_gp.ipynb)
* [The Kernel Cookbook: Advice on Covariance functions](https://www.cs.toronto.edu/~duvenaud/cookbook/)
* [Gaussian Processes for Classification With Python](https://machinelearningmastery.com/gaussian-processes-for-classification-with-python/)
* [An Intuitive Tutorial to Gaussian Processes Regression](https://arxiv.org/pdf/2009.10862.pdf)

### Deep Gaussian Processes

* [Deep Gaussian Processes](http://proceedings.mlr.press/v31/damianou13a.pdf)
* [Deep Gaussian Processes_Pytorch](https://docs.gpytorch.ai/en/v1.5.1/examples/05_Deep_Gaussian_Processes/Deep_Gaussian_Processes.html)
* [Deep convolutional Gaussian processes](https://arxiv.org/abs/1810.03052)


## Uncertainty

* [A Survey of Uncertainty in Deep Neural Networks](https://arxiv.org/abs/2107.03342)

# Machine Learning

## Theory
* [GAN] [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
* [Transformer] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* [Browse State-of-the-Art] (https://paperswithcode.com/sota)
* [Deep Learning Monitor](https://deeplearn.org/)
* [Machine Learning White Board](https://www.yuque.com/books/share/f4031f65-70c1-4909-ba01-c47c31398466?#)

## Blog
* [Lil’Log](https://lilianweng.github.io/)
* [The Berkeley Artificial Intelligence Research Blog](https://bair.berkeley.edu/blog/)

# Reinforcement Learning

## Theory
* [Reinforcement learning: A survey](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
* [A Markovian Decision Process](https://www.jstor.org/stable/24900506?seq=1)
* [Planning and acting in partially observable stochastic domains](https://www.sciencedirect.com/science/article/pii/S000437029800023X)
* [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/pdf/2108.13264.pdf)

## Classic Papers
* [DQN] [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [PPO 2017] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
* [SAC 2018] [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
* [SAC 2018] [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
* [DDPG] [CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING](https://arxiv.org/pdf/1509.02971.pdf)
* [Spinning Up](https://spinningup.openai.com/en/latest/)

## Trustworthy RL
* [Trustworthy Reinforcement Learning Against Intrinsic Vulnerabilities: Robustness, Safety, and Generalizability](https://arxiv.org/pdf/2209.08025.pdf)

## RL Environment Setting
* [gym](https://github.com/openai/gym)
* [highway-env](https://github.com/eleurent/highway-env)
* [SMARTS](https://github.com/huawei-noah/SMARTS)
* [SUMO](https://github.com/eclipse/sumo)
* [metadrive](https://github.com/metadriverse/metadrive)
* [Procgen Benchmark](https://github.com/openai/procgen)
* [DI-drive](https://github.com/opendilab/DI-drive)
* [D4RL,offline RL benchmark](https://arxiv.org/pdf/2004.07219.pdf)
* [reinforcement learning environments you must know](https://medium.com/@mlblogging.k/15-awesome-reinforcement-learning-environments-you-must-know-a38fb75867f2)

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
* [GEM] [Gradient Episodic Memory for Continual Learning](https://arxiv.org/pdf/1706.08840.pdf)
* [A-GEM] [EFFICIENT LIFELONG LEARNING WITH A-GEM](https://arxiv.org/pdf/1812.00420.pdf)
* [PopArt] [Multi-task deep reinforcement learning with popart](https://arxiv.org/pdf/1809.04474.pdf)
* [Task-Agnostic Continual Reinforcement Learning: In Praise of a Simple Baseline](https://arxiv.org/pdf/2205.14495.pdf)
* [Awesome Incremental Learning / Lifelong learning](https://github.com/xialeiliu/Awesome-Incremental-Learning)
* [CoLLAs] [Conference on Lifelong Learning Agents](https://virtual.lifelong-ml.cc/papers.html?session=Conference&filter=keywords)
* [A continual learning survey: Defying forgetting in classification tasks](https://arxiv.org/pdf/1909.08383.pdf)
* [CORA: Benchmarks, Baselines, and Metrics as a Platform for Continual Reinforcement Learning Agents](https://drive.google.com/file/d/1mdqte2xbD6HrP49t9fiRqljU6kvyItb9/view)
* [Efficient Lifelong Learning with A-GEM](https://arxiv.org/pdf/1812.00420.pdf)

## Bayesian Approaches: Continual RL
* [Multi-Task Reinforcement Learning: A Hierarchical Bayesian Approach](http://engr.case.edu/ray_soumya/papers/mtrl-hb.icml07.pdf)
* [Task-Agnostic Online Reinforcement Learning with an Infinite Mixture of Gaussian Processes](https://arxiv.org/pdf/2006.11441.pdf)
* [Overcoming Catastrophic Forgetting with Hard Attention to the Task](https://arxiv.org/pdf/1801.01423.pdf)
* [Continual Learning Based on OOD Detection and Task Masking](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/papers/Kim_Continual_Learning_Based_on_OOD_Detection_and_Task_Masking_CVPRW_2022_paper.pdf)
* [Class-Incremental Learning via Dual Augmentation](https://openreview.net/forum?id=8dqEeFuhgMG)

## Memory-based Continual Learning
* [Information-theoretic Online Memory Selection for Continual Learning](https://openreview.net/forum?id=IpctgL7khPp)
* [Continual Learning Based on OOD Detection and Task Masking](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/papers/Kim_Continual_Learning_Based_on_OOD_Detection_and_Task_Masking_CVPRW_2022_paper.pdf)
* [Overcoming Catastrophic Forgetting with Hard Attention to the Task](https://arxiv.org/pdf/1801.01423.pdf)
* [A Multi-Head Model for Continual Learning via Out-of-Distribution Replay](https://arxiv.org/pdf/2208.09734.pdf)

## Gradient-based Continual Learning
* [Gradient Projection Memory for Continual Learning](https://openreview.net/forum?id=3AOj0RCNC2)
* [Continual Learning with Recursive Gradient Optimization](https://openreview.net/forum?id=7YDLgf9_zgm)
* [TRGP: Trust Region Gradient Projection for Continual Learning](https://openreview.net/forum?id=iEvAf8i6JjO)
* [What Should I Know? Using Meta-Gradient Descent for Predictive Feature Discovery in a Single Stream of Experience](https://virtual.lifelong-ml.cc/poster_6.html)

# Generalization :dart:

## ML generalization
* [Generalizing to Unseen Domains via Adversarial Data Augmentation](https://proceedings.neurips.cc/paper/2018/file/1d94108e907bb8311d8802b48fd54b4a-Paper.pdf)
* [Shuffle Augmentation of Features from Unlabeled Data for Unsupervised Domain Adaptation](https://arxiv.org/pdf/2201.11963.pdf)

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
* [NMER] [Neighborhood Mixup Experience Replay: Local Convex Interpolation for Improved Sample Efficiency in Continuous Control Tasks](https://rmsander.github.io/projects/nmer_tech_report.pdf) [[github]](https://github.com/rmsander/interreplay)
* [Augmix] [AUGMIX: A SIMPLE DATA PROCESSING METHOD TO IMPROVE ROBUSTNESS AND UNCERTAINTY](https://arxiv.org/pdf/1912.02781.pdf)
* [Augmax] [AugMax: Adversarial Composition of Random Augmentations for Robust Training](https://proceedings.neurips.cc/paper/2021/file/01e9565cecc4e989123f9620c1d09c09-Paper.pdf)
* [Continuous Transition] [Continuous Transition: Improving Sample Efficiency for Continuous Control Problems via MixUp](https://arxiv.org/pdf/2011.14487.pdf)
* [S4RL] [S4RL: Surprisingly Simple Self-Supervision for Offline Reinforcement Learning in Robotics](https://arxiv.org/pdf/2103.06326.pdf)
* [Automatic Gaussian Noise] [A simple way to make neural networks robust against diverse image corruptions](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480052.pdf)
* [Certifying Some Distributional Robustness with Principled Adversarial Training](https://arxiv.org/pdf/1710.10571.pdf)

## Model Perspective
* [Sparsity and Heterogeneous Dropout for Continual Learning in the Null Space of Neural Activations](https://arxiv.org/pdf/2203.06514.pdf)
* [Privileged Information Dropout in Reinforcement Learning](https://arxiv.org/pdf/2005.09220.pdf)

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
* [Data-efficient reinforcement learning with self-predictive representations](https://arxiv.org/pdf/2007.05929.pdf)

### RL with GNN
* [Learning Object-Centered Autotelic Behaviors with Graph Neural Networks](https://arxiv.org/pdf/2204.05141.pdf)
* [Hyperbolic Deep Reinforcement Learning](https://arxiv.org/pdf/2210.01542.pdf)
* [Hyperbolic Graph Neural Networks](https://openreview.net/pdf?id=S1eIpVSgUS)

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
* [Generalizing to Unseen Domains via Adversarial Data Augmentation](https://proceedings.neurips.cc/paper/2018/file/1d94108e907bb8311d8802b48fd54b4a-Paper.pdf)

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

# Causal Reasoning 
* [Generalizing Goal-Conditioned Reinforcement Learning with Variational Causal Reasoning](https://arxiv.org/pdf/2207.09081.pdf)
* [Learning Neural Causal Models from Unknown Interventions](https://arxiv.org/pdf/1910.01075.pdf)
* [Learning Neural Causal Models with Active Interventions](https://arxiv.org/pdf/2109.02429.pdf)
* [Experimental design for cost-aware learning of causal graphs](https://arxiv.org/pdf/1810.11867.pdf)
* [Invariant causal prediction for block mdps](https://arxiv.org/pdf/2003.06016.pdf)
* [Learning invariant representations for reinforcement learning without reconstruction](https://arxiv.org/pdf/2006.10742.pdf)

# Robustness
## Adversarial Attack
* [FGSM] [Adversarial Examples: Attacks and Defenses forDeep Learning](https://arxiv.org/pdf/1712.07107.pdf)
* [PGD] [Towards Deep Learning Models Resistant to Adversarial](https://arxiv.org/pdf/1706.06083.pdf)
* [Tactics of Adversarial Attack on Deep Reinforcement Learning Agents](https://williamd4112.github.io/pubs/ijcai2017_adv.pdf)

## Adversarial Robustness
* [DataAug+AdversarialTraining] [Fixing Data Augmentation to Improve Adversarial Robustness](https://arxiv.org/abs/2103.01946)
* [DataAug+AdversarialTraining] [Robustness and Accuracy Could Be Reconcilable by (Proper) Definition](https://arxiv.org/abs/2202.10103)
## RL Robustness
* [Maximum Entropy RL (Provably) Solves Some Robust RL Problems](https://openreview.net/pdf?id=PtSAD3caaA2)

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

# RL Application

## Traffic Problem based on RL
 * [PhD thesis 2022] [Many-Agent Reinforcement Learning: A Modern Approach](https://discovery.ucl.ac.uk/id/eprint/10124273/12/Yang_10124273_thesis_revised.pdf)
 * [PhD thesis 2020] [Safe and Efficient Reinforcement Learning for Behavioural Planning in Autonomous Driving](https://hal.inria.fr/tel-03035705/document)
 * [Efficient Learning of Safe Driving Policy via Human-AI Copilot Optimization][web](https://decisionforce.github.io/HACO/) [paper](https://decisionforce.github.io/HACO/)

### Traffic env setting
* [highway-env] (https://github.com/eleurent/highway-env)
* [SMARTS] (https://github.com/huawei-noah/SMARTS)
* [SUMO] (https://github.com/eclipse/sumo)
* [metadrive] (https://github.com/metadriverse/metadrive)

## CARLA

### Reinforcement Learning CARLA

* [RAIL 2021] [Learning to drive from a world on rails](https://arxiv.org/pdf/2105.00636)

* [PhD thesis 2020] [Safe and Efficient Reinforcement Learning for Behavioural Planning in Autonomous Driving](https://hal.inria.fr/tel-03035705/document)

* [2019] [End-to-End Model-Free Reinforcement Learning for Urban Driving using Implicit Affordances](https://arxiv.org/abs/1911.10868)

### Imitation Learning CARLA

* [LAV 2022] [Learning from All Vehicles](https://arxiv.org/pdf/2203.11934)

* [cheating 2021] [Learning by cheating](https://arxiv.org/pdf/1912.12294)

* [TransFuser 2021] [Multi-Modal Fusion Transformer for End-to-End Autonomous Driving](https://arxiv.org/abs/2104.09224)

### 3D Nerf attack
* [Fooling LiDAR Perception via Adversarial Trajectory Perturbation](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Fooling_LiDAR_Perception_via_Adversarial_Trajectory_Perturbation_ICCV_2021_paper.pdf)
* [Physically Realizable Adversarial Examples for LiDAR Object Detection](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tu_Physically_Realizable_Adversarial_Examples_for_LiDAR_Object_Detection_CVPR_2020_paper.pdf)
* [LG-GAN: Label Guided Adversarial Network for Flexible Targeted Attack of Point Cloud-based Deep Networks](https://arxiv.org/pdf/2011.00566.pdf)
* [Robustness of 3D Deep Learning in an Adversarial Setting](https://arxiv.org/pdf/1904.00923.pdf)
* [Geometric Adversarial Attacks and Defenses on 3D Point Clouds](https://arxiv.org/pdf/2012.05657.pdf)
* [Adversarial Sensor Attack on LiDAR-based Perception in Autonomous Driving](https://arxiv.org/pdf/1907.06826.pdf)
* [Towards Robust LiDAR-based Perception in Autonomous Driving: General Black-box Adversarial Sensor Attack and Countermeasures](https://arxiv.org/pdf/2006.16974.pdf)

## RL Safety
* [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565)

## Robots Application
* [Bayesian Meta-Learning for Few-Shot Policy Adaptation Across Robotic Platforms](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9636628)
* [Data-Efficient Reinforcement and Transfer Learning in Robotics](https://www.diva-portal.org/smash/get/diva2:1501390/FULLTEXT01.pdf)
* [Sim-to-Real Transfer with Incremental Environment Complexity for Reinforcement Learning of Depth-Based Robot Navigation](https://arxiv.org/pdf/2004.14684.pdf)
* [Reinforced Imitation: Sample Efficient Deep Reinforcement Learning for Map-less Navigation by Leveraging Prior Demonstrations](https://arxiv.org/pdf/1805.07095.pdf)
* [Target-Driven Mapless Navigation for Self-Driving Car](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9641134)

### Github Demo
* [Reinforced Imitation: Sample Efficient Deep Reinforcement Learning for Map-less Navigation by Leveraging Prior Demonstrations](https://github.com/ethz-asl/rl-navigation)
* [RL Navigation for Mantis and Turtlebot3](https://github.com/bhctsntrk/mantis_ddqn_navigation)
* [Mobile Robot Planner with Low-cost Cameras Using Deep Reinforcement Learning](https://github.com/trqminh/rl-mapless-navigation)
