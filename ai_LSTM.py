# Importing the libraries
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action): 
        # Inherit the Network object
        super(Network, self).__init__()
        # Declare the input size attribute
        self.input_size = input_size
        # Declare the attribute to store the number of outputs
        self.nb_action = nb_action
        # Defining size of the hidden side
        self.hidden_size = 20
        # Initializing Hidden State and Cell State
        self.cx1 = torch.randn(1, self.hidden_size)
        self.hx1= torch.randn(1, self.hidden_size)
        # Initializing Hidden State and Cell State for LSTM2
        self.cx2 = torch.randn(1, self.hidden_size)
        self.hx2 = torch.randn(1, self.hidden_size)
        # Instantiating the objec that is going to perform a LSTM CELL
        # Inputs : x_t, h_(t-1), c_(t-1)
        # outputs : h_t, c_t
        # -> Input gate, to be element_wise multiplied by gate gate and then summed to the cell state
        # i_t = relu( W_ii*X + B_ii + U_hi*H + B_hi )
        # -> Gate gate, squished to (-1,1) to be multiplied by the input gate and to able to select informations to be passed to cell state
        # g = tanh( W_ig*X + B_ig + U_hg*H + B_hg )
        # -> Forget gate, squished to (-1,1) to be multiplied by the preceding cell state and select wich information is going to be passed to next state
        # f = relu( W_if*X + B_if + U_hf*H + B_hf )
        # -> Cell State, state on this time step that will select wich information will be passed from input to output 
        # c_t = f*c_(t-1) + i_t*g
        # -> Output gate, information from this timestep that will be passed to the output and hidden state pass a multiplication with cell state squished on (-1,1)
        # o = relu( W_io*X + B_io + U_ho*H + B_ho )
        # For better visualization of a LSTM net, access: http://colah.github.io/posts/2015-08-Understanding-LSTMs/  
        self.lstm1 = nn.LSTMCell(self.input_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size,self.nb_action)
        
    # Function that will activate neurons and perform forward propagation to calculate
    # Q-values on each action
    def forward(self, state):
        # Do the foreward pass on the Neural Net
        # LSTM layers
        self.hx1, self.cx1 = self.lstm1(state, (self.hx1, self.cx1))
        self.hx2, self.cx2 = self.lstm2(self.hx1, (self.hx2, self.cx2)) 
        q_values = self.linear(self.hx2)
        return q_values

# Implementing Experience Replay
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.episodeStorer = []

        
    def push(self):
        self.memory.append(self.episodeStorer)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample_episode(self):
        # If we did not stored any episode on the memory yet:
        if len(self.memory) == 0:
            # If there is nothing on the episodeStorer:
            if len(self.episodeStorer) == 0:
                episode = None
            # If there is something on the episodeStorer:
            else:
                episode = self.episodeStorer
        # If there is some episode stored on the memory already:
        else:
            sample_idex = random.sample(range(0, len(self.memory)), 1)[0]
            # Sample_start_idex = random.sample(range(15,50), 1)
            episode = self.memory[sample_idex]
        return episode

# Implementing Deep Q Learning
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        # Gamma attributte is going to be used on the calculation of the Expected Return
        #on Bellman Equation
        self.gamma = gamma
        # List to Store score values to plot in a graph
        self.reward_window = []
        # Istantiating the Network object in an atributte of DQN class
        self.model = Network(input_size, nb_action)
        # Istantiating the ReplayMemory object in an atributte of DQN class
        self.memory = ReplayMemory(100000)
        # Instantiating the Adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.9)
        # Declaring attributes that will be pushed to memory
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        # Temperature object, that will control the exploring/exploiting rate
        self.temperature = 20

    def select_action(self, state):
        # To select an action, we must run the NN to get Q-values for each action for the actual
        # state. After getting Q-values, it is necessary to do a multinomial regression, classifying
        # each Q-value into a probability of being a a good action for that state. To do so, we use function Softmax.
        probs = F.softmax(self.model(Variable(state, volatile = True))*self.temperature)
        # After getting those probs, we sample one of them, using the multinomial method for Torch tensors
        action = probs.multinomial(1)
        # we use .data method to acces the values on the tensors that are wrapped by Variable
        return action.data[0,0]

    def learn(self):
        i = 0
        # Getting an episode from the memory
        episode =self.memory.sample_episode()
        # Defining the graph lenght for a case where len(self.memory.memory) != 0
        graph_end = len(episode) - 1
        if episode is None:
            # If we receive nothing from sample_episode, don't enter on the while loop below
            i = graph_end
        if ( len(self.memory.memory) == 0 ):
            episode = self.memory.episodeStorer
            graph_end = len(self.memory.episodeStorer) - 1
        # Loop to perform backprop through time on the episode selected from the memory
        while i < graph_end :
            # Get important values from each timestep on the episode 
            state, next_state, action, reward = episode[i]
            # Calculate the Q-value for the action taken on the episodes of memory sampled
            # (gather selects the index corespondent to the action taken on the tensor returned by model)
            outputs = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
            # Calculate the Q-value of all possible actions, given the next state and select the bigger to act on a optimal policy Q*
            # Expected action value given optimal policie Q* on (s,a) given next state s'
            next_outputs = self.model(next_state).detach().max(1)[0]
            # Calculate the value of the discounted reward and then the actual Q-value of the action taken
            # Q* = (s,a) = Expected[ next_reward + gamma * max(Q*(s',a')) ]
            target = self.gamma*next_outputs + reward
            # Calculate the loss between the Q-value for the action taken and 
            # the one that should have been taken acordingly to the actual Q-values
            td_loss = F.smooth_l1_loss(outputs, target) 
            # Zero the gradients of the generated graph
            self.optimizer.zero_grad()
            # Do the backwards pass in the generated graph, calculating the grads for each paramether on the NN
            td_loss.backward(retain_graph = True)
            # Aply the optimization step for each value, actualizing the values of the patamethers of the NN
            self.optimizer.step()
            i += 1
        self.model.hx1.detach_()
        self.model.hx2.detach_()
        self.model.cx1.detach_()
        self.model.cx2.detach_()


    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        # Storing information for one episode
        if len(self.memory.episodeStorer) < 39:
            self.memory.episodeStorer.append((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        # if the episode has ended:
        if len(self.memory.episodeStorer) == 39:    
            # Push the episode to the memory, so we can use it in the experience replay
            self.memory.push()
            # Clear the list that stores episodes
            self.memory.episodeStorer = []
        # Backpropagate information through the network with experience replay
        self.learn()            
        # Increase the temperature paramether, controlling the exploration/exploitaition ratioon the model
        if self.temperature < 100 and len(self.memory.memory) > 15:
            self.temperature += (1/50)
            print('temperature = ', self.temperature)
        # Select an action with the select_action method, that uses the Q-values obtained from the Network
        # aplying the softmax, that performs a multinomial logistic regression to select the best action
        action = self.select_action(new_state)
        # Saving values to be stored on memory
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        # Acualising values on the reward window to get feedback from the model
        if len(self.reward_window) > 100:
            del self.reward_window[0]
        return action
        
    # Function to calc the value of the score on the present reward widown
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    #Function to save the actual brain to be used later
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
