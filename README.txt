"Policy Gradient for CartPole-v1"


This project implements the Policy Gradient (REINFORCE) algorithm to train an agent to solve the CartPole-v1 environment. The implementation includes three variations as required by the assignment: a basic version without a baseline, a version with a constant baseline, and a version with a non-constant (learned) baseline using an Actor-Critic model.


== Requirements ==
To run this code, you need Python 3 and the following libraries:
* torch
* gymnasium
* numpy
* matplotlib


You can install them using pip:
pip install torch gymnasium numpy matplotlib


== How to Run the Code ==
The project is divided into three separate Python scripts. After installing the requirements, you can run each experiment from your terminal. Each script will train the model, save the results, and display the required plots.


1. Part 1: CartPole without Baseline
python cartpole1.py
2. Part 2: CartPole with a Changing/Learned Baseline (Actor-Critic)
This implementation uses a learned value function as the baseline.
python cartpole2.py
3. Part 2: CartPole with a Constant Baseline
This implementation uses the mean of an episode's returns as a constant baseline.
python cartpole3.py


== Algorithm Pseudocode ==
The following pseudocode outlines the algorithms implemented in the scripts.


-- 1. REINFORCE Algorithm (Policy Gradient) --
This algorithm is used in cartpole1.py and cartpole3.py.


Algorithm: REINFORCE (Monte-Carlo Policy Gradient)
   1. Initialize the policy network with random weights. This network decides which action to take.


   2. Loop for a fixed number of episodes:


   3.  Play through one full game from start to finish using the current policy, and save all the states, actions, and rewards.

   4.  For every step taken in the game:

   5.  Calculate the total future reward from that step to the end of the game, making future rewards slightly less important than immediate ones (discounting).

   6.  Calculate a "loss" score for the whole episode. This score is higher for actions that led to low future rewards and lower for actions that led to high future rewards. The goal is to encourage good actions.

   7.  Update the policy network's weights to make the actions that led to high future rewards more likely to be chosen again in similar situations.

Note on Baseline Implementation (cartpole3.py):
To improve training stability, we introduce a baseline. Instead of just looking at the raw future reward, we see if it was better or worse than average.


      * First, we calculate the average of all the future rewards from the episode (the baseline).
      * Then, for each action, we calculate the "advantage" by subtracting this average from its future reward.
      * We then update the policy network using this advantage. An action is only encouraged if it was better than the average, and discouraged if it was worse.


-- 2. Actor-Critic with Baseline Algorithm --
This algorithm is implemented in cartpole2.py. It uses a second neural network (the Critic) to provide a more intelligent, learned baseline.


Algorithm: Actor-Critic with Baseline
      1. Initialize two networks:
      * An "Actor" network that decides which action to take (the policy).
      * A "Critic" network that estimates how good a given situation (state) is.


      2. Loop for a fixed number of episodes:


      3.  Play through one full game using the Actor network, saving all states, actions, and rewards.

      4.  Calculate the actual total future reward for every step taken.

      5.  For every step taken in the game:

      6. Ask the "Critic" network to estimate the value of the current state.

      7. Calculate the "advantage" by subtracting the Critic's estimate from the actual future reward. This tells us if the action taken was better or worse than what the Critic expected.

      8. Calculate the "Actor's loss" based on the advantage. If the advantage is positive (the action was better than expected), update the Actor to make that action more likely in the future.

      9. Calculate the "Critic's loss". This is the difference between the Critic's prediction and the actual future reward.

      10. Update the "Actor" network based on its loss.

      11. Update the "Critic" network based on its loss, teaching it to make more accurate predictions of a state's value.


"Policy Gradient for Pong-v5"


This project implements the Policy Gradient (REINFORCE) algorithm to train an agent to play the Atari game Pong (Pong-v5). The implementation includes three variations as required by the assignment: a basic version without a baseline, a version with a constant baseline, and a version with a non-constant (learned) baseline using an Actor-Critic model.


== Requirements ==
To run this code, you need Python 3 and the following libraries:
* torch
* gymnasium
* numpy
* matplotlib
* ale-py
* shimmy


You can install them using pip:
pip install torch gymnasium numpy matplotlib "gymnasium[atari, accept-rom-license]" shimmy


== How to Run the Code ==
The project is divided into three separate Python scripts. After installing the requirements, you can run each experiment from your terminal. Each script will train the model, save the results, and display the required plots.


1. Part 1: Pong without Baseline
python pong1.py
2. Part 2: Pong with a Changing/Learned Baseline (Actor-Critic)
This implementation uses a learned value function as the baseline.
python pong2.py
3. Part 2: Pong with a Constant Baseline
This implementation uses the mean of the batch returns as a constant baseline.
python pong3.py


== Algorithm Pseudocode ==
The following pseudocode outlines the algorithms implemented in the scripts.


-- 1. REINFORCE Algorithm (Policy Gradient) --
This algorithm is used in pong1.py and pong3.py.


Algorithm: REINFORCE (Monte-Carlo Policy Gradient)
   1. Initialize the policy network with random weights. This network decides which action to take.


   2. Loop for a fixed number of training steps:


   3. Play through a small batch of games from start to finish using the current policy, and save all the states, actions, and rewards.

   4. For every step taken in every game in the batch:

   5. Calculate the total future reward from that step to the end of the game, making future rewards slightly less important than immediate ones (discounting).

   6. Calculate a "loss" score for the whole batch. This score is higher for actions that led to low future rewards and lower for actions that led to high future rewards. The goal is to encourage good actions.

   7. Update the policy network's weights to make the actions that led to high future rewards more likely to be chosen again in similar situations.

Note on Baseline Implementation (pong3.py):
To improve training stability, we introduce a baseline. Instead of just looking at the raw future reward, we see if it was better or worse than average.


      * First, we calculate the average of all the future rewards from the entire batch of games (the baseline).
      * Then, for each action, we calculate the "advantage" by subtracting this average from its future reward.
      * We then update the policy network using this advantage. An action is only encouraged if it was better than the batch average, and discouraged if it was worse.


-- 2. Actor-Critic with Baseline Algorithm --
This algorithm is implemented in pong2.py. It uses a second neural network (the Critic) to provide a more intelligent, learned baseline.


Algorithm: Actor-Critic with Baseline
      1. Initialize two networks:
      * An "Actor" network that decides which action to take (the policy).
      * A "Critic" network that estimates how good a given situation (state) is.


      2. Loop for a fixed number of training steps:


      3. Play through a small batch of games using the Actor network, saving all states, actions, and rewards.

      4. Calculate the actual total future reward for every step taken in the batch.

      5. For every step taken in the batch:

      6. Ask the "Critic" network to estimate the value of the current state.

      7. Calculate the "advantage" by subtracting the Critic's estimate from the actual future reward. This tells us if the action taken was better or worse than what the Critic expected.

      8. Calculate the "Actor's loss" based on the advantage. If the advantage is positive (the action was better than expected), update the Actor to make that action more likely in the future.

      9. Calculate the "Critic's loss". This is the difference between the Critic's prediction and the actual future reward.

      10. Update the "Actor" network based on its loss.

      11. Update the "Critic" network based on its loss, teaching it to make more accurate predictions of a state's value.
