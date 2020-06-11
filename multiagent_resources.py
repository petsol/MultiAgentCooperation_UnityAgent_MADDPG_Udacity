import numpy as np 
import os
import torch
import gym
import matplotlib.pyplot as plt 

def plot_episode_scores(scores, target):
    plt.figure(figsize=(13,7))
    score = plt.plot(scores, color='coral', label='Scores')
    avg100 = plt.plot(np.hstack([np.empty((100,)) * np.nan, np.convolve(scores, np.ones((100,))/100, mode='valid')]), color='lime', label='Mean(100)', linewidth=2)
    target_ = plt.plot(np.ones((len(scores),)) * float(target), color='indigo', label='Target')
    
    plt.title("Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    
    plt.legend()
    plt.show()

class OUActionNoise:
    def __init__(self, mu, sigma=0.12, theta=0.06, dt=1e-3, x0=None, decay=0.999):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.decay = decay
        self.reset()
        
    def __call__(self):
        self.decay *= self.decay
        x = (self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape))*self.decay
        
        self.x_prev = x
        
        return x
        #return np.random.normal(size=self.mu.shape) * self.sigma * self.dt
        
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu) 

        
class ExperienceBuffer:
    
    def __init__(self, batch_size, buffer_length, state_size, action_size, _reward_size=1, _isdone_size=1):
        
        self.batch_size = batch_size
        self.buffer_length = buffer_length
        self.total_collected_samples = 0
        self.collected_samples = 0
        
        self.current_states = np.array(np.zeros((buffer_length, state_size  )), dtype=np.float32)
        self.next_states    = np.array(np.zeros((buffer_length, state_size  )), dtype=np.float32)
        self.actions        = np.array(np.zeros((buffer_length, action_size )), dtype=np.float32)
        self.rewards        = np.array(np.zeros((buffer_length, _reward_size)), dtype=np.float32)
        self.isdones        = np.array(np.zeros((buffer_length, _isdone_size)), dtype=np.float32)
        
        self.insertion_index = 0
        
    def insert(self, current_state, action, reward, next_state, isdone):
        
        self.current_states[self.insertion_index] = current_state
        self.next_states[self.insertion_index] = next_state
        self.actions[self.insertion_index] = action
        self.rewards[self.insertion_index] = reward
        self.isdones[self.insertion_index] = float(isdone)
        
        self.total_collected_samples += 1
        self.collected_samples = min(self.total_collected_samples, self.buffer_length)
        
        self.insertion_index = self.total_collected_samples % self.buffer_length
            
            
    def sample(self, indices=None, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size

        indices = np.random.choice(min(self.buffer_length, self.collected_samples), batch_size) if indices is None else indices
        
        return {    'current_states':    self.current_states[indices],
                    'next_states':       self.next_states[indices],
                    'actions':           self.actions[indices],
                    'rewards':           self.rewards[indices],
                    'isdones':           self.isdones[indices],
                    'indices':           indices
               }
    
class Critic(torch.nn.Module):
    
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkdir=''):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.chk_file = os.path.join(chkdir, name+'_ddpg')
        
        # LAYERS
        self.fc1 = torch.nn.Linear(self.input_dims, self.fc1_dims)
        
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        
        #self.bn1 = torch.nn.LayerNorm(self.fc1_dims)

        self.action_value = torch.nn.Linear(self.n_actions, self.fc1_dims)

        self.fc2 = torch.nn.Linear(self.fc1_dims, self.fc2_dims)


        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        #self.bn2 = torch.nn.LayerNorm(self.fc2_dims)


        self.q = torch.nn.Linear(self.fc2_dims, 1)
        f3 = 0.003
        torch.nn.init.uniform_(self.q.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.q.bias.data, -f3, f3)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.to(self.device)
        
    def forward(self, stateX, actionX):

        q_forming_state = self.fc1(stateX)
        #q_forming_state = self.bn1(q_forming_state)
        q_forming_action = self.action_value(actionX)

        q_forming = torch.add(q_forming_state, q_forming_action)
        q_forming = torch.nn.functional.relu(q_forming)
        
        q_forming = self.fc2(q_forming)
        #q_forming = self.bn2(q_forming_state)

        q_forming = torch.nn.functional.relu(q_forming)
        
        q_forming = self.q(q_forming)
        
        return q_forming
        
    def save_checkpoint(self):
        print('saving checkpoint')
        torch.save(self.state_dict(), self.chk_file)
        
    def load_checkpoint(self):
        print('loading checkpoint')
        torch.load_state_dict(torch.load(self.chk_file)) 
        
class Actor(torch.nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkdir=''):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.chk_file = os.path.join(chkdir, name+'_ddpg')
        
        # LAYERS
        self.fc1 = torch.nn.Linear(self.input_dims, self.fc1_dims)
        
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        
        #self.bn1 = torch.nn.LayerNorm(self.fc1_dims)

        self.fc2 = torch.nn.Linear(self.fc1_dims, self.fc2_dims)

        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        #self.bn2 = torch.nn.LayerNorm(self.fc2_dims)

        self.mu = torch.nn.Linear(self.fc2_dims, self.n_actions)
        f3 = 0.003
        torch.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.mu.bias.data, -f3, f3)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.to(self.device)
    
    def forward(self, state):
        actions_forming = self.fc1(state)
        #actions_forming = self.bn1(actions_forming)
        actions_forming = torch.nn.functional.relu(actions_forming)
        
        actions_forming = self.fc2(actions_forming)
        #actions_forming = self.bn2(actions_forming)
        actions_forming = torch.nn.functional.relu(actions_forming)
        
        actions_forming = self.mu(actions_forming)
        actions_forming = torch.tanh(actions_forming)
        
        return actions_forming

    def save_checkpoint(self):
        print('saving checkpoint')
        torch.save(self.state_dict(), self.chk_file)
        
    def load_checkpoint(self):
        print('loading checkpoint')
        self.load_state_dict(torch.load(self.chk_file))     
        
class Agent:
    def __init__(self, alpha_actor, alpha_critic, input_dims, tau, gamma=0.999, n_actions=2, max_size=100000,
                layer1_size=400, layer2_size=300, batch_size=64, team_spirit=0.5):
        
        self.gamma = gamma
        self.tau = tau
        self.memory1 = ExperienceBuffer(batch_size=batch_size, buffer_length=max_size, state_size=input_dims, action_size=n_actions)
        self.memory2 = ExperienceBuffer(batch_size=batch_size, buffer_length=max_size, state_size=input_dims, action_size=n_actions)

        self.batch_size = batch_size
        
        self.actor1 = Actor(alpha_actor, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='Actor1')
        self.actor1_target = Actor(alpha_actor, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='Actor1Target')
        
        self.actor2 = Actor(alpha_actor, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='Actor2')
        self.actor2_target = Actor(alpha_actor, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='Actor2Target')
        
        self.critic1 = Critic(alpha_critic, input_dims*2, layer1_size, layer2_size, n_actions=n_actions*2, name='Critic1')
        self.critic1_target = Critic(alpha_critic, input_dims*2, layer1_size, layer2_size, n_actions=n_actions*2, name='Critic1Target')
        
        self.critic2 = Critic(alpha_critic, input_dims*2, layer1_size, layer2_size, n_actions=n_actions*2, name='Critic2')
        self.critic2_target = Critic(alpha_critic, input_dims*2, layer1_size, layer2_size, n_actions=n_actions*2, name='Critic2Target')
        
        self.noise1 = OUActionNoise(mu=np.zeros(n_actions))
        self.noise2 = OUActionNoise(mu=np.zeros(n_actions))
        
        self.team_spirit = team_spirit

        #print(self.noise1(), self.noise2())
        
        self.project_parameters_to_target(self.actor1, self.actor1_target, tau=1.)
        self.project_parameters_to_target(self.actor2, self.actor2_target, tau=1.)
        self.project_parameters_to_target(self.critic1, self.critic1_target, tau=1.)
        self.project_parameters_to_target(self.critic2, self.critic2_target, tau=1.)
        
    def choose_action(self, state, actor, noise):
        actor.eval()
        
        state = torch.tensor(state, dtype=torch.float).to(actor.device)
        
        with torch.no_grad():
            mu = actor(state).to(actor.device)
        mu_prime = mu + torch.tensor(noise, dtype=torch.float).to(actor.device)
        
        actor.train()
        return np.clip(mu_prime.cpu().detach().numpy(), -1, 1)
    
    def remark(self, memory, current_state, action, reward, next_state, isdone):
        memory.insert(current_state, action, reward, next_state,  isdone)

    def learn(self):
        if self.memory1.collected_samples >= self.batch_size: # doesn't matter which mem we check
            # _ at end means its a batch in learn()
            sampled_data1_ = self.memory1.sample(batch_size=self.batch_size)
            sampled_data2_ = self.memory2.sample(batch_size=self.batch_size, indices=sampled_data1_['indices'])
            
            state1_present_env_     = torch.tensor(sampled_data1_['current_states'], dtype=torch.float).to(self.critic1.device)
            actionset1_present_env_ = torch.tensor(sampled_data1_['actions'],        dtype=torch.float).to(self.critic1.device)
            reward1_future_env_     = torch.tensor(sampled_data1_['rewards'],        dtype=torch.float).to(self.critic1.device)
            state1_future_env_      = torch.tensor(sampled_data1_['next_states'],    dtype=torch.float).to(self.critic1.device)
            isdone1_future_env_     = torch.tensor(sampled_data1_['isdones'],        dtype=torch.float).to(self.critic1.device)
                      
            state2_present_env_     = torch.tensor(sampled_data2_['current_states'], dtype=torch.float).to(self.critic2.device)
            actionset2_present_env_ = torch.tensor(sampled_data2_['actions'],        dtype=torch.float).to(self.critic2.device)
            reward2_future_env_     = torch.tensor(sampled_data2_['rewards'],        dtype=torch.float).to(self.critic2.device)
            state2_future_env_      = torch.tensor(sampled_data2_['next_states'],    dtype=torch.float).to(self.critic2.device)
            isdone2_future_env_     = torch.tensor(sampled_data2_['isdones'],        dtype=torch.float).to(self.critic2.device) # this we don't really need       
            
            actionset1_future_target_  = self.actor1_target(state1_future_env_) 
            actionset2_future_target_  = self.actor2_target(state2_future_env_) 

            stateX_future_env_         = torch.cat((state1_future_env_, state2_future_env_)              , dim=1)
            actionsetX_future_target_  = torch.cat((actionset1_future_target_, actionset2_future_target_), dim=1)

            q1_future_target_          = self.critic1_target(stateX_future_env_, actionsetX_future_target_) 
            q2_future_target_          = self.critic2_target(stateX_future_env_, actionsetX_future_target_) 

            stateX_present_env_        = torch.cat((state1_present_env_, state2_present_env_)        , dim=1)
            actionsetX_present_env_    = torch.cat((actionset1_present_env_, actionset2_present_env_), dim=1)            
            
            q1_present_local_          = self.critic1(stateX_present_env_, actionsetX_present_env_)
            q2_present_local_          = self.critic2(stateX_present_env_, actionsetX_present_env_)
             
            with torch.no_grad(): # we don't want to backpropagate to the critic_target, now do we?
                q1_present_calc_       = reward1_future_env_ + self.gamma * q1_future_target_ * (1. - isdone1_future_env_)
                q2_present_calc_       = reward2_future_env_ + self.gamma * q2_future_target_ * (1. - isdone2_future_env_) # isdone1_future_env_ would suffice
            
            self.critic1.train()
            self.critic2.train()
            
            q1_present_coopcalc_       = (q1_present_calc_ + self.team_spirit * q2_present_calc_) / (1. + self.team_spirit)
            q2_present_coopcalc_       = (q2_present_calc_ + self.team_spirit * q1_present_calc_) / (1. + self.team_spirit)
            
            critic1_loss = torch.nn.functional.mse_loss(q1_present_local_, q1_present_coopcalc_)
            critic2_loss = torch.nn.functional.mse_loss(q2_present_local_, q2_present_coopcalc_)

            self.critic1.optimizer.zero_grad()
            critic1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1)
            self.critic1.optimizer.step()
            
            self.critic2.optimizer.zero_grad()
            critic2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1)
            self.critic2.optimizer.step()
            
            self.critic1.eval()
            self.critic2.eval()
            
            actionset1_present_local_  = self.actor1(state1_present_env_)
            actionset2_present_local_  = self.actor2(state2_present_env_)

            actionsetX1_present_local_ = torch.cat((actionset1_present_local_, actionset2_present_local_.detach()), dim=1)
            actionsetX2_present_local_ = torch.cat((actionset1_present_local_.detach(), actionset2_present_local_), dim=1)          
            
            self.actor1.train()
            self.actor2.train() 
            
            actor1_losspart = -self.critic1(stateX_present_env_, actionsetX1_present_local_).mean()
            actor2_losspart = -self.critic2(stateX_present_env_, actionsetX2_present_local_).mean()
            
            actor1_loss = (actor1_losspart + actor2_losspart.detach() * self.team_spirit) / (1. + self.team_spirit)
            actor2_loss = (actor2_losspart + actor1_losspart.detach() * self.team_spirit) / (1. + self.team_spirit)
            
            self.actor1.optimizer.zero_grad()
            actor1_loss.backward()
            self.actor1.optimizer.step()
            
            self.actor2.optimizer.zero_grad()
            actor2_loss.backward()
            self.actor2.optimizer.step()
            
            self.project_parameters_to_target(self.actor1, self.actor1_target)
            self.project_parameters_to_target(self.actor2, self.actor2_target)
            self.project_parameters_to_target(self.critic1, self.critic1_target)
            self.project_parameters_to_target(self.critic2, self.critic2_target) 
        
    def project_parameters_to_target(self, local, target, tau=None):
        tau = self.tau if tau is None else tau

        for local_param, target_param in zip(local.parameters(), target.parameters()):
            target_param.data.copy_(tau * (local_param.data) + (1. - tau) * target_param.data)  
                    
    def save_models(self):
        for network in [self.actor1, self.actor1_target, self.actor2, self.actor2_target, 
                        self.critic1, self.critic1_target, self.critic2, self.critic2_target]:
            network.save_checkpoint()
        
    def load_models(self):
        for network in [self.actor1, self.actor1_target, self.actor2, self.actor2_target, 
                        self.critic1, self.critic1_target, self.critic2, self.critic2_target]:
            network.load_checkpoint()  