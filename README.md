# Multi-Agent Advisor Q-Learning

Implementation of ADMIRAL-DM, ADMIRAL-DM(AC), and ADMIRAL-AE algorithms for the paper: [Multi-Agent Advisor Q-learning](https://jair.org/index.php/jair/article/view/13445/26794). Paper published in the Journal of Artificial Intelligence Research (JAIR).  

 
## Code structure


- See folder playground for Pommerman environment with the ADMIRAL-DM, ADMIRAL-DM(AC), and ADMIRAL-AE algorithms (results in Section 5.2, 5.3, 5.4, Appendix C)


- See folder admiralaemaze for experiments with the gridmaze environment with the ADMIRAL-AE algorithm (results in Section 5.1)


- See folder admiraldmmaze for experiments with the gridmaze environment with the ADMRIRAL-DM algorithm (results in Section 5.1)   


- See folder pursuitcode for Pursuit environment with the ADMIRAL-DM, ADMIRAL-DM(AC), ADMIRAL-AE and baseline algorithms (Section 5.3) 


- See folder waterworldcode for Waterworld environment with the ADMIRAL-DM, ADMIRAL-DM(AC), ADMIRAL-AE and baseline algorithms (Section 5.3)


### In each of directories, the files most relevant to our research are:


- playground/examples/onevsone/dqnvsadmiraldm.py:  This file contains the code for training and testing the neural network implementation of ADMIRAL-DM algorithm vs DQN.

Similarly other scripts correspond to training and testing against baselines for ADMIRAL-DM.

- playground/examples/onevsone/dqnvsadmiraldmac:  This file contains the code for training and testing the neural network implementation of ADMIRAL-DM(AC) algorithm vs DQN. 

Similarly other scripts correspond to training and testing against baselines for ADMIRAL-DM(AC). 


- playground/pommerman/agents:  This file contains the code for using the algorithmic implementations to create agents in Pommerman. The code for the rule-based advisors is given in the corresponding scripts of this folder.



- pursuitcode/pettingzoosislpursuitDQN.py :  This file contains the code for training and testing the DQN algorithm in the Pursuit SISL environment.

Similarly other scripts correspond to training and execution for other algorithms in our paper. Remember to train the DQN algorithm before the others (trained DQN is used as the advisor for the others).   


- waterworldcode/pettingzoosislwaterworldDDPG.py :  This file contains the code for training and testing the DDPG algorithm in the Waterworld SISL environment.

Similarly other scripts correspond to training and execution for other algorithms in our paper. Remember to train the PPO algorithm before the ADMIRAL-DM(AC) (trained PPO is used as the advisor).

The algorithmic implementations of different algorithms (baselines and ours) is given in the pursuitcode folder for the Pursuit domain and in the waterworldcode folder for the Waterworld domain. 


- playground/examples/onevsone/onevsoneadvisor1admiralae.py:  This file contains the code for training the neural network implementation of ADMIRAL-AE algorithm with the first (best) advisor vs DQN.

Similarly run the onevsoneadvisor2admiralae.py to run the implementation with Advisor2 and so on. 


- admiralaemaze/RL_brain_admiralae_advisor:  This file contains the code for training the ADMIRAL-AE algorithm with the first (best) advisor.

Similarly, look at files RL_brain_admiralae_advisor2, RL_brain_admiralae_advisor3, RL_brain_admiralae_advisor4 for Advisors 2,3 and 4 respectively. 

- admiralaemaze/run_this_advisor.py:   Script to run the ADMIRAL-AE training with the best advisor (Advisor1). 

Similarly, the run_this_advisor2 contains the script to run the Advisor2 and so on. 


Similarly, the corresponding code in the folder admiraldmmaze will run the advisors and tabular version of ADMIRAL-DM algorithm. 


- playground/examples/onevsone: This folder contains the script to train the ADMIRAL-DM and ADMIRAL-DM(AC) algorithms on the Pommerman Domain A - onevsone.

- playground/examples/onevsone/admiraldmvsdeepsarsa_advisor1.py: This file runs the Pommerman onevsone competition for Deep Sarsa vs ADMIRAL-DM with Advisor 1 (best advisor). 

Similarly, admiraldmvsdeepsarsa_advisor2.py runs the competition for Deep Sarsa vs ADMIRAL-DM with Advisor2 and so on. 


Similarly, the folder playground/examples/teamcompetition - contains all the scripts for the Pommerman team domain.
 


### Algorithmic Implementation 


- playground/example/onevsone/RL_brain_admiralae.py :  This file contains the code for the algorithmic implementation of ADMIRAL-AE. 

- playground/example/onevsone/RL_brain_admrialdmac.py :  This file contains the code for the algorithmic implementation of ADMIRAL-DM(AC). 

- playground/example/onevsone/RL_brain_admiraldm.py :  This file contains the code for the algorithmic implementation of ADMIRAL-DM. 

- playground/example/onevsone/RL_brain_CHAT.py :  This file contains the code for the algorithmic implementation of CHAT. 

- playground/example/onevsone/RL_brain_DQN.py :  This file contains the code for the algorithmic implementation of DQN. 

- playground/example/onevsone/DQfD_V3.py :  This file contains the code for the algorithmic implementation of DQfD. 

- playground/example/onevsone/RL_brain_Deepsarsa.py :  This file contains the code for the algorithmic implementation of Deep Sarsa. 

- waterworldcode/ppo.py :  This file contains the code for the algorithmic implementation of PPO. 

- waterworldcode/ddpg.py :  This file contains the code for the algorithmic implementation of DDPG. 
 

## Installation Instructions for Ubuntu 18.04



### Pommerman 

##### Requirements

Atleast 

- `python==3.6.1`


```shell
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6
```



- `Tkinter`

```shell
sudo apt-get update
sudo apt-get install python3-tk
```


- `tensorflow 2`

```shell
pip install --upgrade pip
pip install tensorflow
```

- `pandas`

```shell
pip install pandas
```
- `matplotlib`

```shell
pip install matplotlib
```

Download the files and store them in a separate directory to install packages from the requirements file. 

```shell
cd playground
pip install -U . 
```


For more help with the installation, look at the instructions in [Playground](https://github.com/MultiAgentLearning/playground). 

Now you can just run the respective files mentioned in the above section to run our code.


For the Pursuit and Waterworld domains you also need to install petting zoo library. Note that we use version 1.5.0 of pettingzoo for the Pursuit domain and 1.4.2 for the Waterworld domain. Please install the appropriate version using the command below. 

### Petting Zoo

- `gym` (Version 0.18.0)

```shell
pip install gym==0.18.0
```

- `pettingzoo` (Version 1.4.2) 

```shell
pip install pettingzoo==1.4.2
```


Now, you can just run the relevant files mentioned in the above section to run our code. 


## Note

This is research code and will not be actively maintained. Please send an email to ***s2ganapa@uwaterloo.ca*** for questions or comments. 



## Code Citations

We would like to cite [Playground](https://github.com/MultiAgentLearning/playground) for code in the playground folder. The files for running Pommerman game are retained from this repository, with only the algorithms implemented from our side as described in the code structure here. 
We would also like to cite [Reinforcement Learning and Tensorflow](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow), from which the basic structure of the grid world game has been obtained. Further, some of our algorithmic implementations and baselines are based on the implementation in this repository. We thank the [Petting Zoo](https://github.com/PettingZoo-Team/PettingZoo) repository for providing the Pursuit and Waterworld environments. We would also like to thank the [go2sea](https://github.com/go2sea/DQfD) repository on which our DQfD baseline implementation is based. 


## Paper citation

If you found this helpful, please cite the following paper:

<pre>
@article{Subramanian2022multiagent,
  title={Multi-Agent Advisor Q-Learning},
  author={Subramanian, Sriram Ganapathi and Taylor, Matthew E and Larson, Kate and Crowley, Mark},
  journal={Journal of Artificial Intelligence Research},
  volume={74},
  pages={1--74},
  year={2022}
}
</pre>


