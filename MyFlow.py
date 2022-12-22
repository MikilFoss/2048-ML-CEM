#custom machine learning framework
import numpy as np
import random
from tqdm import tqdm
import statistics
import matplotlib.pyplot as plt
import time
from multiprocessing import Process, Queue
from IPython.display import clear_output

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x):
    return np.maximum(0.01 * x, x)


class Network:
    def __init__(self, input_size, output_size, structure, activations):
        self.input_size = input_size
        self.output_size = output_size
        self.structure = structure
        self.weights = []
        self.biases = []
        self.layers = []
        self.layer_sizes = [input_size] + structure + [output_size]
        self.activations = [None] + activations + [sigmoid]
        for i in range(len(self.layer_sizes) - 1):
            self.weights.append(np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i]))
            self.biases.append(np.random.randn(self.layer_sizes[i + 1], 1))
            self.layers.append(np.zeros((self.layer_sizes[i + 1], 1)))
        self.layers.append(np.zeros((self.output_size, 1)))

    def __delattr__(self, __name: str) -> None:
        pass
    def calculate(self, input):
        t = []
        for i in range(len(input)):
            t.append([input[i]])
        input = np.array(t)
        del(t)
        self.layers[0] = input
        for i in range(len(self.weights)):
            self.layers[i + 1] = self.activations[i + 1](np.dot(self.weights[i], self.layers[i]) + self.biases[i])
        return sum(self.layers[-1].tolist(), [])

class PooledNetwork:
    def __init__(self, input_size, output_size, structure, activations):
        self.input_size = input_size
        self.output_size = output_size
        self.structure = structure
        self.weights = []
        self.layers = []
        self.layer_sizes = [input_size] + structure + [output_size]
        self.activations = [None] + activations + [sigmoid]
        self.weights.append(np.random.randn(self.layer_sizes[0], self.layer_sizes[1]))
        self.layers.append(np.zeros((input_size, 1)))
        for i in range(1, len(self.layer_sizes) - 1):
            self.weights.append(np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i+1] + self.layer_sizes[i] - 1))
            self.layers.append(np.zeros((self.layer_sizes[i + 1], 1)))
        self.layers.append(np.zeros((self.output_size, 1)))
    def calculate(self, input, iterations):
        t = []
        for i in range(len(input)):
            t.append([input[i]])
        input = np.array(t)
        del(t)
        self.layers[0] = input
        counter = 0
        for i in range(1,len(self.layer_sizes) - 1):
            #calculate the pooled layer
            self.layers[i] = self.activations[i](np.dot(self.weights[i-1][:self.layer_sizes[i]], self.layers[i-1]) + self.biases[i])
            for j in range(iterations[counter]):
                self.layers[i] = self.activations[i](np.dot(self.weights[i][self.layer_sizes[i]:], self.layers[i]), axis=0)
            counter += 1
        self.layers[-1] = self.activations[-1](np.dot(self.weights[-1], self.layers[-2]))

class Agent:
    def __init__(self, Enviornment, shape, activations,max_steps):
        self.env = Enviornment
        self.network = Network(self.env.input_size, self.env.num_actions, shape, activations)
        self.ep_return = 0
        self.score = 0
        self.max_steps = max_steps
    
    def play_game(self, render=False):
        self.ep_return = 0
        self.steps = 0
        isover, state, reward = self.env.reset()
        while not isover:
            if render:
                self.env.render()
            actionprobs = self.network.calculate(state)
            isover, state, reward = self.env.step(actionprobs)
            self.ep_return = reward
            self.steps += 1
            if self.steps > self.max_steps:
                break
        self.score = self.ep_return
        return self.ep_return


class GeneticTrainer:
    def __init__(self,env, agentCount, agentShape, agentActivations, max_steps, mutationRate, elitism, maxGenerations, gameCount, existing=None):
        self.env = env
        self.agentCount = agentCount
        self.agentShape = agentShape
        self.agentActivations = agentActivations
        self.max_steps = max_steps
        self.mutationRate = mutationRate
        self.elitism = elitism
        self.maxGenerations = maxGenerations
        self.gameCount = gameCount
        self.agents = []
        self.num_threads = 4
        if existing is None:
            for i in range(self.agentCount):
                self.agents.append(Agent(self.env, self.agentShape, self.agentActivations, self.max_steps))
        else:
            self.agents = existing


    def train(self,threaded, freq=10, render=0):
        rewards = []
        for i in tqdm(range(self.maxGenerations)):

            #agents play the games and a median score is calculated
            st = time.time()
            if threaded:
                self.play_games_threaded()
            else:
                self.play_games()
            et = time.time()
            #sort the agents within the list accourding to their score
            self.sort_agents()
            if i % freq == 0:
                clear_output()
                print("Generation: ", i, " Top Score: ", self.agents[0].score, " Time: ", et-st)

            #TODO add render
            
            rewards.append(self.agents[0].score)
            #reproduce the agents
            self.reproduce()
            #print reward metrics for the generation
            #print("time to play", et-st)
        plt.plot(rewards)
        plt.ylabel('max reward')
        return self.agents
    
    def play_games(self):
        #single threaded
        for agent in self.agents:
            scores = []
            for i in range(self.gameCount):
                scores.append(agent.play_game())
            agent.score = statistics.median(scores)
    
    
    def play_games_threaded(self):
        i = 0
        processes = []
        firstindex = 0
        lastindex = len(self.agents)//self.num_threads
        return_queue = Queue()
        while lastindex < len(self.agents):
            ray = [firstindex,lastindex]
            p = Process(target=self.play_game, args=(ray,return_queue,))
            processes.append(p)
            p.start()
            firstindex = lastindex
            lastindex += len(self.agents)//self.num_threads
        if lastindex != len(self.agents):
            ray = [firstindex,len(self.agents)]
            p = Process(target=self.play_game, args=(ray,return_queue,))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
        
        while not return_queue.empty():
            i, score = return_queue.get()
            self.agents[i].score = score

    def play_game(self, indexes, return_queue):
        
        for agent in self.agents[indexes[0]:indexes[1]]:
            scores = []
            for i in range(self.gameCount):
                scores.append(agent.play_game())
            agent.score = statistics.median(scores)
            return_queue.put((self.agents.index(agent),agent.score))
        
        #return_queue.put(indexes,[self.agents[indexes[0]:indexes[1]]])
        
        

    def sort_agents(self):
        self.agents.sort(key=lambda x: x.score, reverse=True)

    def reproduce(self):
        #elitism
        newAgents = []
        for i in range(self.elitism):
            newAgents.append(self.agents[i])
        #reproduction
        for i in range(self.elitism, self.agentCount):
            #select parents
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            #crossover
            child = self.crossover(parent1, parent2)
            #mutation
            self.mutate(child)
            #add child to new agents
            newAgents.append(child)
        self.agents = newAgents
    
    def select_parent(self):
        #roulette wheel selection
        totalScore = sum([agent.score for agent in self.agents])
        if totalScore == 0:
            return random.choice(self.agents)
            print("total score is 0")
        randomScore = random.uniform(0, totalScore)
        currentScore = 0
        for agent in self.agents:
            currentScore += agent.score
            if currentScore > randomScore:
                return agent


    def crossover(self, parent1: Agent, parent2: Agent):
        child = Agent(self.env, self.agentShape, self.agentActivations, self.max_steps)
        if type(child) != type(parent1) or type(child) != type(parent2):
            print(type(child))
            print(type(self.agents[0]))
        for i in range(len(child.network.weights)):
            child.network.weights[i] = np.where(np.random.rand(*child.network.weights[i].shape) < 0.5, parent1.network.weights[i], parent2.network.weights[i])
            child.network.biases[i] = np.where(np.random.rand(*child.network.biases[i].shape) < 0.5, parent1.network.biases[i], parent2.network.biases[i])
        return child

    def mutate(self, agent):

        for i in range(len(agent.network.weights)):
            for j in range(len(agent.network.weights[i])):
                for k in range(len(agent.network.weights[i][j])):
                    if random.random() < self.mutationRate:
                        agent.network.weights[i][j][k] = random.uniform(-1, 1)
            for j in range(len(agent.network.biases[i])):
                for k in range(len(agent.network.biases[i][j])):
                    if random.random() < self.mutationRate:
                        agent.network.biases[i][j][k] = random.uniform(-1, 1)
    