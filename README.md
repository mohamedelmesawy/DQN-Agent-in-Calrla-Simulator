# DQN-Agent in Carla Simulator - Self Driving Car

<div align="center" id="top"> 
 
</div>

## About ##

~~~
- CARLA provides open digital assets (urban layouts, buildings, vehicles) that were created for this purpose and can be used freely. The simulation platform supports flexible specification of sensor suites, environmental conditions, full control of all static and dynamic actors, maps generation, and much more.

- This project shows how to train a Self-Driving Car agent by segmenting the RGB Camera image streaming from an urban scene in the CARLA simulator to simulate the autonomous behavior of the car inside the game, utilizing the Reinforcement Learning technique Deep Q-Network [DQN].
~~~


## Reinforcement Learning: ##
<img src="https://user-images.githubusercontent.com/28452932/152051120-87a299ce-3939-4e42-8f08-70ccf78dc139.png" width="1000">

~~~
- A reinforcement learning task is about training an  AGENT  which interacts with its  ENVIRONMENT. 
- The agent arrives at different scenarios known as STATES by performing  ACTIONS. 
- Actions lead to REWARDS which could be positive or negative.
- The agent has only one purpose here – to maximize its total reward across an EPISODE. 
- This episode is anything and everything that happens between the first state and the last or terminal state within the environment. 
- We reinforce the agent to learn to perform the BEST ACTIONS by experience.  This is the STRATEGY or POLICY.
~~~



### Q-Learning:
~~~
- Q-Learning builds a Q-table of State-Action values, with dimension (s, a), where s is the number of states and a is the number of actions. 
- The Q-table maps state and action pairs to a Q-value.
- However, in a real-world scenario, the number of states could be huge, making it computationally intractable to build a table.
~~~

<img src="https://user-images.githubusercontent.com/28452932/152043019-578b0f79-ce9f-4411-882f-ac0782a78c2e.png" width="1000">
<img src="https://user-images.githubusercontent.com/28452932/152046607-374184ff-84ec-4432-af23-df56780742fd.png" width="1000">


### Deep Q-Learning and DQN:
~~~
- In deep Q-learning, we use a Neural Network to approximate the Q-value function, which is called a Deep Q Network [DQN].
- This function maps a state [Input] to the Q-values of all the actions that can be taken from that state [Output].
~~~

<img src="https://user-images.githubusercontent.com/28452932/152043059-ef7a93ea-0376-4c40-98e0-f072edd16de5.png" width="1000">
<img src="https://user-images.githubusercontent.com/28452932/152053640-d91d7882-7813-4101-a147-fd18bb5058e5.png" width="1000">



## DQN Agent inside CARLA Simulator ##

![image](https://user-images.githubusercontent.com/28452932/152042082-b74282a2-a590-4c62-be9a-40c66e167f30.png)

~~~
DQN Algorithm: [1] Feed the DQN Agent with the preprocessed segmented urban scene image (state s) 
                   - and it returns the Q-values of all possible actions in the state [different values for throttle, steer].
               [2] Select an Action using the Epsilon-Greedy Policy. 
                   - With the probability epsilon, we select a random action a.
                   - With the probability 1-epsilon, we select an action that has a maximum Q-value, such as a = argmax(Q(s,a,w)).
               [3] Perform this Action in a state s and move to a new state s’ to receive a reward. 
                   - This state s’ is the next image. 
                   - We store this transition in our Replay Buffer as <s,a,r,s’>
               [4] Next, sample some Random Batches of transitions from the Replay Buffer.
               [5] Calculate the Loss  which is just the squared difference between target-Q and predicted-Q.
               [6] Perform Gradient Descent with respect to our actual network parameters [DQN Agent] in order to minimize this loss.
               [7] After every C iteration, copy our actual network weights to the Target network weights.
               [8] Repeat these steps for M number of episodes.
~~~



## Features ##
    
:heavy_check_mark: Training the model. \
:heavy_check_mark: Evaluatation using pretrained model.


## Technologies ##

The following tools you will need to run this repo:

- [Python](https://www.python.org/)
- [Carla 1.13](https://github.com/carla-simulator/carla/releases/)
- [Tensorflow](https://www.tensorflow.org/install/)


## Requirements ##

Before starting, you need to have [Git](https://git-scm.com) and [Anaconda](https://www.anaconda.com/) installed.
Also, you will need to run each python command in different terminal as they will be running at the same time during the training.

```bash
# Clone this project
$ git clone git@github.com:mohamedelmesawy/DQN-Agent-in-Calrla-Simulator

# Create Conda Environment
$ conda env create -f carla.yml
$ conda activate carla

# Run the Carla Simulator - default port 2000
$ ./CarlaUE4.sh -quality-level Low -fps 20
or  
$ ./CarlaUE4.exe -quality-level Low -fps 20

# Generate Vehicles and Walkers inside Carla, [-n] for num_vehicles and [-w] for num_walkers
$ python ./generate_traffic.py -n 70 -w 100

# Run the DQN-Agent at Test Mode using the pretrained model
$ python ./loadmodel.py

# Run the DQN-Agent at Train Mode
$ python ./DQN_in_Carla.py

# Check the updated logs
$ tensorboard --logdir logs
```


By RAM-Team: <a href="https://github.com/mohamedelmesawy" target="_blank">MOhamed ElMesawy</a>, <a href="https://github.com/Afnaaan" target="_blank">Afnan Hamdy</a>, <a href="https://github.com/Rawan-97" target="_blank">Rowan ElMahalawy</a>, <a href="https://github.com/alihasanine" target="_blank">Ali Hassanin</a>, and <a href="https://github.com/MSamaha91" target="_blank">MOhamed Samaha</a>
&#xa0; 

## Acknowledgements ##
This code was built on this repo [CARLA RL](https://github.com/Sentdex/Carla-RL) and this book [Hands-On Reinforcement Learning with Python](https://github.com/PacktPublishing/Hands-On-Reinforcement-Learning-with-Python). 


<a href="#top">Back to top</a>
