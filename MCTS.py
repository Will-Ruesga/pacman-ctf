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
    def __init__(self, state, agentIndex, parent=None, action=None, depth=np.inf, agent= None):
        self.state = state
        self.agentIndex = agentIndex
        self.agent=agent
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
    
    def bestChild(self, c=10.0):
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
        child = Node(nextState, self.agentIndex, parent=self, action=action, depth=self.depth, agent=self.agent)
        self.children.append(child)

        return child
    
    def defaultPolicy(self):
        state = self.state.deepCopy()
        agentIndex = self.agentIndex
        i = 0
        initScore = self.agent.getScore(state)
        while (not state.isOver()) and i < self.depth:
            actions = state.getLegalActions(agentIndex)
            action = np.random.choice(actions)
            state = state.generateSuccessor(agentIndex, action)
            i+=1
        
        return (self.agent.getScore(state) - initScore)*10 + self.rewardFunction(state)
    
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
    
    def rewardFunction(self, state, mode="attack"):
        # Get the position and carrying status of the agent
        agent_pos = state.getAgentPosition(self.agentIndex)
        is_pacman = state.getAgentState(self.agentIndex).isPacman
        carrying = state.getAgentState(self.agentIndex).numCarrying

        # Get the number of opponents scared and caught
        opponents_scared = sum([1 for opp in self.agent.getOpponents(state) if state.getAgentState(opp).scaredTimer > 0])
        opponents_caught = sum([1 for opp in self.agent.getOpponents(state) if state.getAgentState(opp).isPacman and state.getAgentState(opp).numCarrying == 0])
        opponents_carrying = sum([state.getAgentState(opp).numCarrying for opp in self.agent.getOpponents(state)])

        distance_to_own_food = min([self.agent.getMazeDistance(agent_pos, food_pos) for food_pos in self.agent.getFood(state).asList()])+1
        distance_to_opponents = min([self.agent.getMazeDistance(agent_pos, state.getAgentPosition(opp)) for opp in self.agent.getOpponents(state)])+1

        # Compute the reward based on the weighting factors
        w0, w1, w2, w3, = self.getWeights(mode)

        return (w0 / distance_to_own_food) + w1 * carrying + w2 * opponents_carrying + (w3 / distance_to_opponents)
    
    def getWeights(self, mode):
        if mode == "attack":
            w0 = 1.0  # Weight for distance to own food
            w1 = 2.0  # Weight for food carried
            w2 = -1.0  # Weight for opponents carrying
            w3 = -0.2  # Weight for distance to opponents (min)
        return w0, w1, w2, w3
    
    def printRewards(self):
        print(" ")
        print("Agent Index: " + str(self.agentIndex) + " | Agent position: " + str(self.state.getAgentPosition(self.agentIndex)) + " | Carrying: " + str(self.state.getAgentState(self.agentIndex).numCarrying))
        print("---- ACTIONS: REWARDS / VISITS / AVERAGE ----")
        for child in self.children:
            print(child.action + ": " + str(child.totalReward) + " / " + str(child.visits) + " / " + str(child.totalReward/self.visits))

class MCTSPacman(CaptureAgent):
    def __init__(self, index, depth=20, timeForComputing=0.9):
        CaptureAgent.__init__(self, index)
        self.timeForComputing = timeForComputing
        self.depth=depth

    def registerInitialState(self, state):
        CaptureAgent.registerInitialState(self, state)
        self.start = state.getAgentPosition(self.index)
        self.prevAction = None
        
    def chooseAction(self, initState):
        # Set the start state
        state = initState.deepCopy()
        
        # Create the root node
        rootNode = Node(state, self.index, depth=self.depth, action=self.prevAction, agent=self)
        
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

        self.prevAction = rootNode.bestChild(c=0).action
        rootNode.printRewards()
        # Return the best action
        return rootNode.bestChild(c=0).action