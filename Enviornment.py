import random
import numpy as np
import game
import time
#if both numbers are same then do 0 else do 1
class Space2048:

    def __init__(self):
        self.num_actions = 4
        self.input_size = 16
        
    #reset the environment, return [isover,state,reward]

    #take in the output of the network, move that action, return [isover,state,reward]

    def reset (self):
        self.ep_return  = 0
        self.currentgame = game.start()
        return [False, sum(self.currentgame.logmoves(),[]), 0]

    def step(self,actionprobs):
        action = np.argmax(actionprobs)
        changed = False

        if action==0:
                #move up
            changed = self.currentgame.moveUp()
        elif action==1:
                #move down
            changed = self.currentgame.moveDown()
        elif action==2:
                #move right
            changed = self.currentgame.moveRight()
        elif action==3:
                #move left
            changed = self.currentgame.moveLeft()
        if changed:
            self.currentgame.spawn()
        
        self.currentgame.getState()
        self.ep_return = max(sum(self.currentgame.state,[]))*10 + sum(sum(self.currentgame.state,[]))
        if changed == False:
            return [True, sum(self.currentgame.logmoves(),[]), self.ep_return]
        return [self.currentgame.isover, sum(self.currentgame.logmoves(),[]), self.ep_return]
    
    def render(self):
        print(self.currentgame.state)
        time.wait(0.1)