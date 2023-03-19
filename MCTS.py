# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import math
import time
import numpy as np

from game import Directions
from captureAgents import CaptureAgent

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'MCTSPacman', second = 'MCTSPacman'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class Node:
    def __init__(self, state, agentIndex, parent=None, action=None, depth=np.inf):
        self.state = state
        self.agentIndex = agentIndex
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.totalReward = 0
        self.untriedActions = self.state.getLegalActions(agentIndex)
        self.depth = depth
        self.improve()

    def improve(self):
        if not self.action is None:
            if self.action == Directions.STOP:
                self.totalReward -= 1

        if not self.parent is None:
            if not self.parent.action is None:
                #self.untriedActions = np.delete(self.untriedActions, np.where(self.untriedActions == Directions.REVERSE[self.parent.action]))
                if self.action == Directions.REVERSE[self.parent.action]:
                    self.totalReward -= 1
                if self.parent.action == self.action:
                    self.totalReward += 1
            
    def isLeaf(self):
        return len(self.children) == 0
    
    def isTerminal(self):
        return self.state.isOver()
    
    def isFullyExpanded(self):
        return len(self.untriedActions) == 0
    
    def bestChild(self, c=1.0):
        bestScore = float('-inf')
        bestChild = None
        
        for child in self.children:
            exploitation = child.totalReward / child.visits
            exploration =  (2*math.log(self.visits) / child.visits) ** 0.5
            score = exploitation + c * exploration
            
            if score > bestScore:
                bestScore = score
                bestChild = child
        
        return bestChild
    
    def expand(self):
        action = np.random.choice(self.untriedActions)
        self.untriedActions.remove(action) #np.delete(self.untriedActions, np.where(self.untriedActions == action))
        nextState = self.state.generateSuccessor(self.agentIndex, action)
        child = Node(nextState, self.agentIndex, parent=self, action=action, depth=self.depth)
        self.children.append(child)

        return child
    
    def defaultPolicy(self):
        state = self.state.deepCopy()
        agentIndex = self.agentIndex
        i = 0
        initScore = state.getScore()
        while (not state.isOver()) and i < self.depth:
            actions = state.getLegalActions(agentIndex)
            action = np.random.choice(actions)
            state = state.generateSuccessor(agentIndex, action)
            i+=1
        
        return (state.getScore() - initScore)*10
    
    def treePolicy (self):
        node = self
        while not node.isTerminal():
            if not node.isFullyExpanded():
                return node.expand()
            else:
                node = node.bestChild()
        return node
    
    def backup(self, reward):
        self.visits += 1
        self.totalReward += reward
        
        if self.parent is not None:
            self.parent.backup(reward)
    
    def reward_function(self, game_state, agent):
        reward = 0
        # Reward the agent for capturing the opponent's flag
        if agent.get_position() == game_state.get_opponent_flag_position() and agent.has_flag():
            reward += 100
        # Punish the agent for getting tagged by the opponent
        elif agent.get_position() in game_state.get_opponent_positions():
            reward -= 10
        # Encourage the agent to eat dots and power pellets
        elif agent.get_position() in game_state.get_pellet_positions():
            reward += 1
        return reward
    
    def printRewards(self):
        print("---- ACTIONS: REWARDS ----")
        for child in self.children:
            print(child.action + ": " + str(child.totalReward))

class MCTSPacman(CaptureAgent):
    def __init__(self, index, depth=100, timeForComputing=.9):
        CaptureAgent.__init__(self, index)
        self.timeForComputing = timeForComputing
        self.depth=depth

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)
        self.prevAction = None
        
    def chooseAction(self, gameState):
        # Set the start state
        state = gameState.deepCopy()
        
        # Create the root node
        rootNode = Node(state, self.index, depth=self.depth, action=self.prevAction)
        
        nodes=0
        # Start the MCTS algorithm
        startTime = time.time()
        while time.time() - startTime < self.timeForComputing:
            node = rootNode

            # Select - Tree Policy
            node = node.treePolicy()

            # Simulate - Default Policy
            reward = node.defaultPolicy()

            # Backup
            node.backup(reward)
            nodes+=1

        self.prevAction = rootNode.bestChild(c=0).action
        rootNode.printRewards()
        print(nodes)
        # Return the best action
        return rootNode.bestChild(c=0).action