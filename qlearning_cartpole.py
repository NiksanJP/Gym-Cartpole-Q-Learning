import numpy as np 
import pandas as pd
import gym

class agent:
    def __init__(self, actions):
        self.learningRate = 0.1
        self.epsilon = 0.9
        self.discountRate = 0.9
        self.action = actions
        try:
            self.table = self.readPanda()
            print(self.table)
        except:
            self.table = self.createTable()
        
        self.decimalRoundups = 12 
    
    def createTable(self):
        table = pd.DataFrame(columns=self.action, dtype=np.float64)
        return table
    
    def chooseAction(self, observation):
        observation = str(([round(observation[0],self.decimalRoundups), round(observation[1],self.decimalRoundups), round(observation[2],self.decimalRoundups), round(observation[3],3)]))
        self.checkObservationExistance(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.action)
        return action
    
    def learn(self, state, action, reward, new_state):
        state = str(([round(state[0],self.decimalRoundups), round(state[1],self.decimalRoundups), round(state[2],self.decimalRoundups), round(state[3],self.decimalRoundups)]))
        new_state = str(([round(new_state[0],self.decimalRoundups), round(new_state[1],self.decimalRoundups), round(new_state[2],self.decimalRoundups), round(new_state[3],self.decimalRoundups)]))
        
        self.checkObservationExistance(str(new_state))
        
        predict = self.table.loc[state, action]

        target = reward + self.discountRate * self.table.loc[new_state, :].max()
        self.table.loc[state, action] += self.learningRate * (target - predict)
        
    def checkObservationExistance(self, observation):
        if str(observation) not in self.table.index:
            self.table = self.table.append(
                pd.Series(
                    [0]*len(self.action),
                    index=self.table.columns,
                    name = str(observation)
                )
            )
            
    def savePanda(self):
        df = pd.DataFrame.copy(self.table)
        df.to_csv('cartpole.csv', index=False)
        
    def readPanda(self):
        return pd.read_csv('cartpole.csv')
            
        

env = gym.make("CartPole-v1")
observation = env.reset()
totalReward = 0

agent = agent((0,1))

for episode in range(100):
    agent.savePanda()
    print("SAVED")
    if totalReward == 1000:
        print("TOTAL EPISODES : ", episode)
        print("TOTAL REWARD : ", totalReward)
        break
    totalReward = 0
    for steps in range(1000):
        reward = 0
        interaction = 'Episode %s: total_steps = %s Reward = %s' % (episode+1, steps, reward)
        env.render()
        
        if episode == 0 and steps == 0:
            action = env.action_space.sample()
        else:
            action = agent.chooseAction(prevObservation)
        
        observation, reward, done, info = env.step(action)
        
        if episode == 0 and steps == 0:
            prevObservation = observation
        
        
        totalReward += reward
        
        agent.learn(prevObservation, action, reward, observation)
        
        prevObservation = observation
        
        if done:
            observation = env.reset()
            agent.savePanda()
            totalReward = 0
        

env.close()