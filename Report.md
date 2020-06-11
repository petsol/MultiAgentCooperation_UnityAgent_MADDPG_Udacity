## Introduction

This report details the implementation and hyperparameters used for the model used to solve the Tennis environment. The Tennis Environment was solved (reaching a mean score of 0.5 in the last 100 episodes) using a version of MADDPG algorithm in episode 794.

## Implementation of the MADDPG algorithm

The MADDPG algorithm uses a Multi Actor-Critic, Local-Target network structure. The Actors are the main focus of the training, at inference, only the policies contained in the Actor(-Local)s will be used. The Actor is in itself a continuous Proximal Policy Optimization algorithm, which takes the State as an input and transfers the data to the actual action-set that is a set (or a single) of continuous values (in this case the racket movement values). To aide the optimization process a Critic network is used, which is in itself a Deep Q Learning algorithm, trying to approximate the Q function for continuous state-action pairs. However in the MADDPG algorithm both (in this case there are two) Actors are trained using both Critics. In this particular implementation a hyperparameter called team spirit drives the difference between the two Critic networks. The inputs for both critic networks are both Agents’ states and both Agents’ actions. However when the expected Q value is calculated (using the Bellman equation) the Q value part for the “other” agent is weighted by the team spirit value (when team spirit is 1. the contribution is equal, when 0. it is omitted). In the current network this is done by weighted averaging the two Agents’ Q expected value. This results in two Critic networks that both consider the other Actors’ rewards, states and actions, but in different proportions. As in the DDPG algorithm the optimization of the Critic networks are done via the target Actor and target Critic obtained expected values (through Bellman eq.) MSE compared to the Critic output itself. The Actors’ optimization is done via both Critic networks using the Actors input (and states). In this specific implementation team spirit is also used to weight the inputs. The Critic’s output that was optimized to correspond primarily to the specific Actor is weighted higher in that Actor’s loss function (again if team spirit is 1. they both contribute equally). Thus the inference of the other agents actions (without knowing its states) is implemented, but with a weighted impact on the acting output. After these operations, the target networks are soft updated with a tau coefficient at each learning step. Both optimizers are ADAM, with fix learning rate. The implemented MADDPG algorithm also employs two ExperienceBuffers. They are interlinked by parallel insertions and first random index sampling passed to drive second memory sampling. To generate exploratory behavior a random noise is added to the Actors’ output during training, this Noise is a product of Ornstein-Uhlenbeck process, a highly correlative noise, and can be considered in itself as a random walk in continuous time. However in this case a noise decay is also implemented, to ensure that the primarily beneficial exploratory behavior tied to the noise doesn’t interfere in later stages to appropriately “hit” the ball. 

## Implementation details

Batch normalization was not used in this project (code is commented out in the network sections). The Actor-network has an input layer of state size (24), two hidden layers with 128  and 64 neurons respectively and an output layer of action-set size (2). The transfer functions except the output layer are all ReLU. The output layer has tangent-hyperbolic transfer function to correspond to the action intervals [-1, 1]. The Critic Network has similar structure except for an extra layer for action inputs, that is joined into the first hidden layer by direct addition to the input layers feed for each hidden layer neuron and ReLU-ed together before entering the first hidden layer. The output of the Critic network is a single neuron with a linear transfer function (no transfer function). The hidden layers are initialized in both networks with zero mean uniform distribution dependent on layer size.

The training parameters were learning rates as 0.001 for the actor and 0.001 for the critic network. The soft update was applied at every learning step with a tau of 0.01. A batch size of 256 was used for training and for the critic network gamma was set at 0.97. Choosing a gamma over 0.95 was relatively straightforward because of the sparse distanced reward structure. The experience buffer size was 100,000.

An interesting mistake made showed the power of joined training. First by mistake the two experience buffers were sampled with independent random index selection, thus with different states, rewards etc. Retrospectively it seems strange that the model even converged, but it did at episode 1727. After this was corrected the target was reached more than twice as fast. 

The hyperparameter tuning was relatively simple, the model converged immediately after implementation of noise decay. However the model converged much faster with a relatively higher 0.75 team spirit value, but even higher values slowed convergence down. The noise parameters were sigma = 0.12, theta 0.06 and dt=0.001. Noise decay was set at 0.999. 

Although theoretically I talked about two agents, in practice the implementation used a single Agent object that handled the learning of both agents.

## Results

The above model with the mentioned hyperparameters, was able to converge to the desired target at episode 794, with a value of 0.5079.

![Continuous Control Convergence Graph](https://github.com/petsol/MultiAgentCooperation_UnityAgent_MADDPG_Udacity/blob/master/Tennis_scores.png?raw=true)

Udacity sources and Phil Tabor’s ‘LunarLanderContinuous-v2’ DDPG video was used as a model for the DDPG implementation.

## Future improvements

At the beginning of the training the models catch up slowly. This is understandable and most probably the consequence of the relatively low off chance to randomly hit the ball. In some cases not only initial improvements were delayed by several hundreds of epochs, but also in some cases one agent underperformed relative to the other for a few hundred episodes before catching up. This could be equalized by higher but faster and nonlinearly decaying noise and also by employing Prioritized Experience Replay to resample the few rewarded instances in the beginning phase. 
I also considered a third Critic network, that would be a state-value network and would respond evenly to the absolute state (both agents and ball state). The Actors would be optimized for a second time with this Critic equally.



Sources:
Udacity Deep Reinforcement Learning Nanodegree
- Ornstein–Uhlenbeck process:
   https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
- Udacity Deep Reinforcement Learning Git Repository:
   https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet 
- Reinforcement Learning in Continuous Action Spaces | DDPG Tutorial (Pytorch) (Phil Tabor)
   https://www.youtube.com/watch?v=6Yd5WnYls_Y&t=2208s
- Multi-Agent Deep Deterministic Policy Gradient (MADDPG) https://www.youtube.com/watch?v=Ku5h_FBL6Lg



