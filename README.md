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
  * [Goal Conditioned RL](#goal-conditioned-rl)
  * [xRL](#xRL)
  
* [Pretrain Reinforcement Learning](#Pretrain-reinforcement-learning)

* [Offline Reinforcement Learning](#offline-reinforcement-learning)

* [Continual Reinforcement Learning](#continual-reinforcement-learning)
  * [Conferences](#conferences) 

* [Continual Learning](#continual-learning)
  * [Regularization-based Approaches](#Regularization-based-approaches)
  * [Memory-based Approaches](#memory-based-approaches)
  * [Gradient-based Approaches](#Gradient-based-approaches)
  * [Multi-task Learning](#Multi-task-learning)

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
  * [Poisoning Attack](#Poisoning-attack)
  * [Backdoor Attack](#Backdoor-attack)
 
* [Exploration Strategy](#exploration-strategy)
  * [Count-based Exploration Strategy](#Count-based-exploration-strategy)
  * [Prediction-based Exploration Strategy](#Prediction-based-Exploration-Strategy)
  * [Unsupervised Active Pretraining](#Unsupervised-Active-Pretraining)

* [RL Application](#RL-Application)
  * [Traffic Problem based on RL](#traffic-problem-based-on-RL)
    * [Traffic env setting](#traffic-env-setting)
   
  * [Trajectory Generation](#Trajectory-generation)

  * [CARLA](#carla)
    * [Reinforcement Learning CARLA](#reinforcement-learning-carla)
    * [Imitation Learning CARLA](#imitation-learning-carla)
    * [3D/Nerf attack at Autonomous Driving](#3D-Nerf-attack-at-Autonomous-Driving)
    * [3D Point Cloud attack at Autonomous Driving](#3D-Point-Cloud-attack-at-Autonomous-Driving)

  * [RL Safety](#RL-SAFETY)
  * [Robots Appilcation](#robots-application)
    * [Github Demo](#github-demo)

* [Large Model](#Large-Model)
  * [Large Language Model](#Large-Language-Model)
  * [Large Vision Language Model](#Large-Vision-Language-Model)
 


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

## Goal Conditioned RL
* [Discovering Generalizable Spatial Goal Representations via Graph-based Active Reward Learning](https://proceedings.mlr.press/v162/netanyahu22a/netanyahu22a.pdf)
* [A Relational Intervention Approach for Unsupervised Dynamics Generalization in Model-Based Reinforcement Learning](https://arxiv.org/pdf/2206.04551.pdf)
* [Generalizable Imitation Learning from Observation via Inferring Goal Proximity](https://openreview.net/pdf?id=lp9foO8AFoD)
* [Learning Goal-Conditioned Policies Offline with Self-Supervised Reward Shaping](https://openreview.net/pdf?id=8tmKW-NG2bH)
* [Rethinking goal-conditioned supervised learning and its connection to offline rl](https://arxiv.org/pdf/2202.04478.pdf)
* [Don’t Change the Algorithm, Change the Data: Exploratory Data for Offline Reinforcement Learning](https://arxiv.org/pdf/2201.13425.pdf)
* [Offline Goal-Conditioned Reinforcement Learning via f-Advantage Regression](https://arxiv.org/pdf/2206.03023.pdf)
* [Prioritized offline Goal-swapping Experience Replay](https://arxiv.org/pdf/2302.07741.pdf)
* [Swapped goal-conditioned offline reinforcement learning](https://arxiv.org/pdf/2302.08865v1.pdf)
* [Learning to Reach Goals via Iterated Supervised Learning](https://openreview.net/pdf?id=rALA0Xo6yNJ)
* [Imitating Graph-Based Planning with Goal-Conditioned Policies](https://openreview.net/pdf?id=6lUEy1J5R7p)
* [A Relational Intervention Approach for Unsupervised Dynamics Generalization in Model-Based Reinforcement Learning](https://arxiv.org/pdf/2206.04551.pdf)

## xRL
* [Awesome Explainable Reinforcement Learning](https://github.com/Plankson/awesome-explainable-reinforcement-learning#Reward-Explaining)
* [Trainify: A CEGAR-Driven Training and Verification Framework for Safe Deep Reinforcement Learning](https://faculty.ecnu.edu.cn/_upload/article/files/39/62/197880be44aba90d9d44ac6de8bb/b7ef9fd1-51e0-4284-8af0-5d7a2f9f1869.pdf)
* [Robust Deep Reinforcement Learning against Adversarial Perturbations on State Observations](https://proceedings.neurips.cc/paper/2020/file/f0eb6568ea114ba6e293f903c34d7488-Paper.pdf)

# Pretrain Reinforcement Learning
* [smart](https://github.com/microsoft/smart)

# Offline Reinforcement Learning
* [D4RL: DATASETS FOR DEEP DATA-DRIVEN REINFORCEMENT LEARNING](https://arxiv.org/pdf/2004.07219.pdf)
* [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/pdf/2006.04779.pdf)
* [Off-Policy Deep Reinforcement Learning without Exploration](https://arxiv.org/pdf/1812.02900.pdf)
* [General Batch RL papers](https://github.com/apexrl/Batch-Offline--RL-Paper-Lists#general-batch-rl )

# Continual Reinforcement Learning
* [Continual World: A Robotic Benchmark For Continual Reinforcement Learning](https://arxiv.org/pdf/2105.10919.pdf) [website](https://sites.google.com/view/continualworld)
* [CORA: Benchmarks, Baselines, and Metrics as a Platform for Continual Reinforcement Learning Agents](https://drive.google.com/file/d/1mdqte2xbD6HrP49t9fiRqljU6kvyItb9/view)
* [SEQUOIA: A SOFTWARE FRAMEWORK TO UNIFY CONTINUAL LEARNING RESEARCH](https://arxiv.org/pdf/2108.01005.pdf)
* [Task-Agnostic Continual Reinforcement Learning: In Praise of a Simple Baseline](https://arxiv.org/pdf/2205.14495.pdf)
* [Task-Agnostic Online Reinforcement Learning with an Infinite Mixture of Gaussian Processes](https://arxiv.org/pdf/2006.11441.pdf)
* [Model-Free Generative Replay for Lifelong Reinforcement Learning: Application to Starcraft-2](https://virtual.lifelong-ml.cc/poster_50.html)
* [Self-Activating Neural Ensembles for Continual Reinforcement Learning](https://virtual.lifelong-ml.cc/poster_31.html)
* [Reactive Exploration to Cope With Non-Stationarity in Lifelong Reinforcement Learning](https://virtual.lifelong-ml.cc/poster_35.html)
* [Modular Lifelong Reinforcement Learning via Neural Composition](https://openreview.net/forum?id=5XmLzdslFNN)
* [Lifelong Policy Gradient Learning of Factored Policies for Faster Training Without Forgetting](https://arxiv.org/pdf/2007.07011.pdf)

## Conferences
* [CoLLAs] [Conference on Lifelong Learning Agents](https://virtual.lifelong-ml.cc/papers.html?session=Conference&filter=keywords)
* [Awesome Incremental Learning / Lifelong learning](https://github.com/xialeiliu/Awesome-Incremental-Learning)


# Continual Learning
* [A continual learning survey: Defying forgetting in classification tasks](https://arxiv.org/pdf/1909.08383.pdf)

## Regularization-based Approaches
* [L2/EWC] [Overcoming catastrophic forgetting inneural networks](https://www.pnas.org/doi/epdf/10.1073/pnas.1611835114)
* [VCL] [VARIATIONAL CONTINUAL LEARNING](https://arxiv.org/pdf/1710.10628.pdf)
* [MAS] [Memory Aware Synapses: Learning what (not) to forget](https://arxiv.org/pdf/1711.09601.pdf)

## Memory-based Approaches
* [Information-theoretic Online Memory Selection for Continual Learning](https://openreview.net/forum?id=IpctgL7khPp)
* [Continual Learning Based on OOD Detection and Task Masking](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/papers/Kim_Continual_Learning_Based_on_OOD_Detection_and_Task_Masking_CVPRW_2022_paper.pdf)
* [Overcoming Catastrophic Forgetting with Hard Attention to the Task](https://arxiv.org/pdf/1801.01423.pdf)
* [A Multi-Head Model for Continual Learning via Out-of-Distribution Replay](https://arxiv.org/pdf/2208.09734.pdf)
* [Overcoming Catastrophic Forgetting with Hard Attention to the Task](https://arxiv.org/pdf/1801.01423.pdf)
* [Continual Learning Based on OOD Detection and Task Masking](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/papers/Kim_Continual_Learning_Based_on_OOD_Detection_and_Task_Masking_CVPRW_2022_paper.pdf)
* [Class-Incremental Learning via Dual Augmentation](https://openreview.net/forum?id=8dqEeFuhgMG)
* [Continual Adaptation of Visual Representations via Domain Randomization and Meta-learning](https://arxiv.org/pdf/2012.04324.pdf)

## Gradient-based Approaches
* [GEM] [Gradient Episodic Memory for Continual Learning](https://arxiv.org/pdf/1706.08840.pdf)
* [A-GEM] [Efficient Lifelong Learning with A-GEM](https://arxiv.org/pdf/1812.00420.pdf)
* [Improved Schemes for Episodic Memory-based Lifelong Learning](https://proceedings.neurips.cc/paper/2020/file/0b5e29aa1acf8bdc5d8935d7036fa4f5-Paper.pdf)
* [Gradient Projection Memory for Continual Learning](https://openreview.net/forum?id=3AOj0RCNC2)
* [Continual Learning with Recursive Gradient Optimization](https://openreview.net/forum?id=7YDLgf9_zgm)
* [TRGP: Trust Region Gradient Projection for Continual Learning](https://openreview.net/forum?id=iEvAf8i6JjO)
* [What Should I Know? Using Meta-Gradient Descent for Predictive Feature Discovery in a Single Stream of Experience](https://virtual.lifelong-ml.cc/poster_6.html)
* [Few-Shot Learning by Dimensionality Reduction in Gradient Space](https://arxiv.org/pdf/2206.03483.pdf)
* [Improving Task-free Continual Learning by Distributionally Robust Memory Evolution](https://proceedings.mlr.press/v162/wang22v/wang22v.pdf)

## Isolation-based Approaches
* [PackNet] [PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://arxiv.org/pdf/1711.05769.pdf)

## Multi-task learning
* [PopArt] [Multi-task deep reinforcement learning with popart](https://arxiv.org/pdf/1809.04474.pdf)
* [Multi-Task Reinforcement Learning: A Hierarchical Bayesian Approach](http://engr.case.edu/ray_soumya/papers/mtrl-hb.icml07.pdf)
* [Multi-Task Reinforcement Learning with Context-based Representations](https://arxiv.org/pdf/2102.06177.pdf)

# Generalization :dart:

## ML generalization
* [Generalizing to Unseen Domains via Adversarial Data Augmentation](https://proceedings.neurips.cc/paper/2018/file/1d94108e907bb8311d8802b48fd54b4a-Paper.pdf)
* [Shuffle Augmentation of Features from Unlabeled Data for Unsupervised Domain Adaptation](https://arxiv.org/pdf/2201.11963.pdf)
* [Invariant Risk Minimization](https://arxiv.org/pdf/1907.02893.pdf)
* [Out-of-Distribution Generalization via Risk Extrapolation](https://arxiv.org/pdf/2003.00688.pdf)

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
* [Generalization of Reinforcement Learning with Policy-Aware Adversarial Data Augmentation](https://openreview.net/pdf?id=rzDUUiEFiG)

## Data Augmentation (state-based)
* [mixup] [mixup: BEYOND EMPIRICAL RISK MINIMIZATION](https://arxiv.org/pdf/1710.09412.pdf)
* [mixreg] [Improving Generalization in Reinforcement Learning with Mixture Regularization](https://arxiv.org/pdf/2010.10814v1.pdf)
* [NMER] [Neighborhood Mixup Experience Replay: Local Convex Interpolation for Improved Sample Efficiency in Continuous Control Tasks](https://rmsander.github.io/projects/nmer_tech_report.pdf) [[github]](https://github.com/rmsander/interreplay)
* [Interpolated Experience Replay for Improved Sample Efficiency of Model-Free Deep Reinforcement Learning Algorithms](https://dspace.mit.edu/bitstream/handle/1721.1/138972/Sander-rmsander-meng-eecs-2021-thesis.pdf?sequence=1&isAllowed=y)
* [Augmix] [AUGMIX: A SIMPLE DATA PROCESSING METHOD TO IMPROVE ROBUSTNESS AND UNCERTAINTY](https://arxiv.org/pdf/1912.02781.pdf)
* [Augmax] [AugMax: Adversarial Composition of Random Augmentations for Robust Training](https://proceedings.neurips.cc/paper/2021/file/01e9565cecc4e989123f9620c1d09c09-Paper.pdf)
* [Continuous Transition] [Continuous Transition: Improving Sample Efficiency for Continuous Control Problems via MixUp](https://arxiv.org/pdf/2011.14487.pdf)
* [S4RL] [S4RL: Surprisingly Simple Self-Supervision for Offline Reinforcement Learning in Robotics](https://arxiv.org/pdf/2103.06326.pdf)
* [Automatic Gaussian Noise] [A simple way to make neural networks robust against diverse image corruptions](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480052.pdf)
* [Certifying Some Distributional Robustness with Principled Adversarial Training](https://arxiv.org/pdf/1710.10571.pdf)
* [Augmentation with Projection: Towards an Effective and Efficient Data Augmentation Paradigm for Distillation](https://openreview.net/forum?id=kPPVmUF6bM_)
* [Adversarial Examples Can Be Effective Data Augmentation for Unsupervised Machine Learning](https://arxiv.org/pdf/2103.01895.pdf)
* [Disentangling Adversarial Robustness and Generalization](https://arxiv.org/pdf/1812.00740.pdf)
* [MaxUp: Lightweight Adversarial Training with Data Augmentation Improves Neural Network Training](https://openaccess.thecvf.com/content/CVPR2021/papers/Gong_MaxUp_Lightweight_Adversarial_Training_With_Data_Augmentation_Improves_Neural_Network_CVPR_2021_paper.pdf)
* [Generalization of Reinforcement Learning with Policy-Aware Adversarial Data Augmentation](https://openreview.net/pdf?id=rzDUUiEFiG)

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
* [Grid-to-Graph: Flexible Spatial Relational Inductive Biases for Reinforcement Learning](https://arxiv.org/pdf/2102.04220.pdf)

## IL generalization

* [arXiv 2021] [Generalization Guarantees for Imitation Learning](https://arxiv.org/pdf/2008.01913)

# Domain Generalization (DG)

* [Domain Generalization: A Survey](https://arxiv.org/abs/2103.02503)
* [Source‑Guided Adversarial Learning and Data Augmentation for Domain Generalization](https://link.springer.com/content/pdf/10.1007/s42979-020-00375-w.pdf?pdf=button)

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
* [Boosting Adversarial Attacks with Momentum](https://arxiv.org/pdf/1710.06081.pdf)

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
* [A Relational Intervention Approach for Unsupervised Dynamics Generalization in Model-Based Reinforcement Learning](https://arxiv.org/pdf/2206.04551.pdf)

# Robustness
## Adversarial Attack
* [FGSM] [Adversarial Examples: Attacks and Defenses for Deep Learning](https://arxiv.org/pdf/1712.07107.pdf)
* [PGD] [Towards Deep Learning Models Resistant to Adversarial](https://arxiv.org/pdf/1706.06083.pdf)
* [Tactics of Adversarial Attack on Deep Reinforcement Learning Agents](https://williamd4112.github.io/pubs/ijcai2017_adv.pdf)
* [Adversarial Attacks on Neural Network Policies](https://arxiv.org/pdf/1702.02284.pdf)
* [Boosting Adversarial Attacks with Momentum](https://arxiv.org/pdf/1710.06081.pdf)
* [DELVING INTO TRANSFERABLE ADVERSARIAL EXAMPLES AND BLACK-BOX ATTACKS](https://arxiv.org/pdf/1611.02770.pdf)

## Adversarial Robustness
* [DataAug+AdversarialTraining] [Fixing Data Augmentation to Improve Adversarial Robustness](https://arxiv.org/abs/2103.01946)
* [DataAug+AdversarialTraining] [Robustness and Accuracy Could Be Reconcilable by (Proper) Definition](https://arxiv.org/abs/2202.10103)
## RL Robustness
* [Maximum Entropy RL (Provably) Solves Some Robust RL Problems](https://openreview.net/pdf?id=PtSAD3caaA2)
* [PhD thesis 2022] [Towards Verifiable, Generalizable and Efficient Robust Deep Neural Networks](https://infoscience.epfl.ch/record/295825)

## Poisoning Attack
* [Adaptive Reward-Poisoning Attacks against Reinforcement Learning](https://arxiv.org/pdf/2003.12613.pdf)
* [Provably Efficient Black-Box Action Poisoning Attacks Against Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2021/file/678004486c119599ed7d199f47da043a-Paper.pdf)
* [Vulnerability-Aware Poisoning Mechanism for Online RL with Unknown Dynamics](https://arxiv.org/pdf/2009.00774.pdf)

## Backdoor Attack
* [Adversarial Skill Learning for Robust Manipulation](https://arxiv.org/pdf/2011.03383.pdf)
* [Robotic Control in Adversarial and Sparse Reward Environments: A Robust Goal-Conditioned Reinforcement Learning Approach](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10018434)

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
 * [Language Instructed Reinforcement Learning for Human-AI Coordination](https://arxiv.org/pdf/2304.07297.pdf) [code](https://github.com/hengyuan-hu/instruct-rl)
 * [The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/pdf/2309.07864v1.pdf)
 * [Self-Refined Large Language Model as Automated Reward Function Designer for Deep Reinforcement Learning in Robotics](https://arxiv.org/pdf/2309.06687.pdf)https://arxiv.org/pdf/2106.09110.pdf
 * [Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory](https://arxiv.org/pdf/2305.17144.pdf)
 * [ProAgent: Building Proactive Cooperative AI with Large Language Models](https://arxiv.org/pdf/2308.11339v2.pdf)
 * [Language Instructed Reinforcement Learning for Human-AI Coordination](https://openreview.net/pdf?id=CSAAs2QAyW)
 * [MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning](https://arxiv.org/pdf/2109.12674.pdf)
 * [Instruct2Act: Mapping Multi-modality Instructions to Robotic Actions with Large Language Model](https://arxiv.org/pdf/2305.11176.pdf)
 * [Instruction-Following Agents with Jointly Pre-Trained Vision-Language Models](https://openreview.net/forum?id=U0jfsqmoV-4)
 * [LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action](https://arxiv.org/pdf/2207.04429.pdf)
 * [Navigation with Large Language Models: Semantic Guesswork as a Heuristic for Planning](https://openreview.net/forum?id=PsV65r0itpo)
 * [Language to Rewards for Robotic Skill Synthesis](https://language-to-reward.github.io/assets/l2r.pdf)
 * [Robotics Toolformer: Synergizing Robotics Reasoning and Acting with Mixed-Modal Tools](https://openreview.net/pdf?id=g7oTwH5-M0)
 * [ViNT: A Large-Scale, Multi-Task Visual Navigation Backbone with Cross-Robot Generalization](https://openreview.net/pdf?id=-K7-1WvKO3F)
 * [GUARDED POLICY OPTIMIZATION WITH IMPERFECT ONLINE DEMONSTRATIONS](https://openreview.net/pdf?id=O5rKg7IRQIO)
 * [Towards Scalable Coverage-Based Testing of Autonomous Vehicles](https://openreview.net/pdf?id=Q9ezhChqnL)
 * [Reinforcement Learning by Guided Safe Exploration](https://arxiv.org/pdf/2307.14316.pdf)
 * [Safe Reinforcement Learning Using Advantage-Based Intervention](https://arxiv.org/pdf/2106.09110.pdf)

### Traffic env setting
* [highway-env] (https://github.com/eleurent/highway-env)
* [SMARTS] (https://github.com/huawei-noah/SMARTS)
* [SUMO] (https://github.com/eclipse/sumo)
* [metadrive] (https://github.com/metadriverse/metadrive)

## Trajectory Generation
* [ScenarioNet: Open-Source Platform for Large-Scale
Traffic Scenario Simulation and Modeling](https://arxiv.org/pdf/2306.12241.pdf)
* [SafeBench: A Benchmarking Platform for Safety
Evaluation of Autonomous Vehicles](https://arxiv.org/pdf/2206.09682.pdf)
* [Drive Like a Human: Rethinking Autonomous Driving with Large Language Models](https://arxiv.org/pdf/2307.07162.pdf)

## CARLA

### Reinforcement Learning CARLA

* [RAIL 2021] [Learning to drive from a world on rails](https://arxiv.org/pdf/2105.00636)

* [PhD thesis 2020] [Safe and Efficient Reinforcement Learning for Behavioural Planning in Autonomous Driving](https://hal.inria.fr/tel-03035705/document)

* [2019] [End-to-End Model-Free Reinforcement Learning for Urban Driving using Implicit Affordances](https://arxiv.org/abs/1911.10868)

### Imitation Learning CARLA

* [LAV 2022] [Learning from All Vehicles](https://arxiv.org/pdf/2203.11934)

* [cheating 2021] [Learning by cheating](https://arxiv.org/pdf/1912.12294)

* [TransFuser 2021] [Multi-Modal Fusion Transformer for End-to-End Autonomous Driving](https://arxiv.org/abs/2104.09224)

### 3D Nerf attack at Autonomous Driving
* [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/pdf/2003.08934.pdf)
* [ViewFool: Evaluating the Robustness of Visual Recognition to Adversarial Viewpoints](https://arxiv.org/pdf/2210.03895.pdf)

### 3D Point Cloud attack at Autonomous Driving
* [Survey] [SoK: On the Semantic AI Security in Autonomous Driving](https://arxiv.org/pdf/2203.05314.pdf)
* [3D object detection on nuscenes](https://paperswithcode.com/sota/3d-object-detection-on-nuscenes)
* [BEVFusion-3D object detection](https://github.com/mit-han-lab/bevfusion)
* [Adversarial Attacks against LiDAR Semantic Segmentation in Autonomous Driving](https://dl.acm.org/doi/pdf/10.1145/3485730.3485935)
* [Fooling LiDAR Perception via Adversarial Trajectory Perturbation](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Fooling_LiDAR_Perception_via_Adversarial_Trajectory_Perturbation_ICCV_2021_paper.pdf)
* [Physically Realizable Adversarial Examples for LiDAR Object Detection](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tu_Physically_Realizable_Adversarial_Examples_for_LiDAR_Object_Detection_CVPR_2020_paper.pdf)

* [Geometric Adversarial Attacks and Defenses on 3D Point Clouds](https://arxiv.org/pdf/2012.05657.pdf)
* [Adversarial Sensor Attack on LiDAR-based Perception in Autonomous Driving](https://arxiv.org/pdf/1907.06826.pdf)
* [Towards Robust LiDAR-based Perception in Autonomous Driving: General Black-box Adversarial Sensor Attack and Countermeasures](https://arxiv.org/pdf/2006.16974.pdf)
* [LG-GAN: Label Guided Adversarial Network for Flexible Targeted Attack of Point Cloud-based Deep Networks](https://arxiv.org/pdf/2011.00566.pdf)
* [Robustness of 3D Deep Learning in an Adversarial Setting](https://arxiv.org/pdf/1904.00923.pdf)
* [AdvSim: Generating Safety-Critical Scenarios for Self-Driving Vehicles](https://arxiv.org/pdf/2101.06549.pdf)
* [Generating Useful Accident-Prone Driving Scenarios via a Learned Traffic Prior](https://openaccess.thecvf.com/content/CVPR2022/papers/Rempe_Generating_Useful_Accident-Prone_Driving_Scenarios_via_a_Learned_Traffic_Prior_CVPR_2022_paper.pdf)

## RL Safety
* [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565)
* [Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer](https://arxiv.org/pdf/2207.14024v3.pdf)

## Robots Application
* [Bayesian Meta-Learning for Few-Shot Policy Adaptation Across Robotic Platforms](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9636628)
* [Data-Efficient Reinforcement and Transfer Learning in Robotics](https://www.diva-portal.org/smash/get/diva2:1501390/FULLTEXT01.pdf)
* [Sim-to-Real Transfer with Incremental Environment Complexity for Reinforcement Learning of Depth-Based Robot Navigation](https://arxiv.org/pdf/2004.14684.pdf)
* [Reinforced Imitation: Sample Efficient Deep Reinforcement Learning for Map-less Navigation by Leveraging Prior Demonstrations](https://arxiv.org/pdf/1805.07095.pdf)
* [Target-Driven Mapless Navigation for Self-Driving Car](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9641134)

## Sim2Real Robots
* [Off-Dynamics Reinforcement Learning: Training for Transfer with Domain Classifiers](https://github.com/google-research/google-research/tree/master/darc)

### Github Demo
* [Reinforced Imitation: Sample Efficient Deep Reinforcement Learning for Map-less Navigation by Leveraging Prior Demonstrations](https://github.com/ethz-asl/rl-navigation)
* [RL Navigation for Mantis and Turtlebot3](https://github.com/bhctsntrk/mantis_ddqn_navigation)
* [Mobile Robot Planner with Low-cost Cameras Using Deep Reinforcement Learning](https://github.com/trqminh/rl-mapless-navigation)
* 

# Large Model
## Large Language Model
* [A Survey of Safety and Trustworthiness of Large Language Models through the Lens of Verification and Validation](https://arxiv.org/pdf/2305.11391.pdf)

## Large Vision Language Model
* [Visual Instruction Tuning](https://arxiv.org/pdf/2304.08485.pdf)
* [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020.pdf)
