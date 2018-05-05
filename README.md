# Learning to Flap using RL techniques

## Implementations:

 - Simple Q-Learning  
 - Q-Learning (with ε-greedy policy) 
 - Deep Q-Network

## Installation Dependencies:

 - Python 3 
 - pygame 
 - scikit-image 
 - Keras 2

## Simple Q-Learning:
### Running the saved model:

    python flappy_rl.py Run

### Training the model:

    python flappy_rl.py Train
To train a fresh model, delete the file **qvalues.txt** before executing the above command.

## Q-Learning (with ε-greedy policy):
### Running the saved model:

    python flappy_rl.py Run greedy

### Training the model:

    python flappy_rl.py Train greedy
To train a fresh model, delete the file **qvalues_greedy.txt** and then execute the above command.

## Deep Q-Network (DQN):
### Running the saved model:

    python dqn.py Run

### Training the model:

    python dqn.py Train
To train a fresh model, delete the file **dqn.h5** and then execute the above command.

## Raw Environment:
https://github.com/sourabhv/FlapPyBird
