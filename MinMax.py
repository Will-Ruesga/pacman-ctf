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
import util
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, copy
from game import Directions
import game
from util import nearestPoint

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
  return [eval(first)(firstIndex, mode="attack"), eval(second)(secondIndex, mode="defense")]

##########
# Agents #
##########

class Node:
    def __init__(self, state, agentIndex, index,  parent=None, action=None, depth=np.inf, agent= None, mode="attack", myTurn=True):
        self.state = state
        self.agentIndex = agentIndex
        self.index = index
        self.agent = agent
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.totalReward = 0
        self.untriedActions = self.state.getLegalActions(index)
        self.depth = depth
        self.mode = mode
        #self.improve()
        self.myTurn=myTurn

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
    
    def bestChild(self, c=10000.0):
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
        nextState = self.state.generateSuccessor(self.index, action)
        child = Node(nextState, self.agentIndex, (self.index+1) % nextState.getNumAgents(), parent=self, action=action, depth=self.depth, agent=self.agent, mode=self.mode, myTurn=(not self.myTurn))
        self.children.append(child)

        return child
    
    def defaultPolicy(self):
        state = self.state.deepCopy()
        agentIndex = self.index
        i = 0
        initScore = copy.deepcopy(self.agent.getScore(state))
        while (not state.isOver()) and i < self.depth:
            actions = state.getLegalActions(agentIndex)
            action = np.random.choice(actions)
            state = state.generateSuccessor(agentIndex, action)
            agentIndex = (agentIndex + 1) % state.getNumAgents()
            i+=1
        
        return self.rewardFunction(state, initScore)
    
    def treePolicy (self, d=1000):
        node = self
        i=0
        while not node.isTerminal() and i <d:
            if not node.isFullyExpanded():
                return node.expand()
            else:
                node = node.bestChild()
            i+=1
        return node
    
    def backup(self, reward):
        self.visits += 1
        self.totalReward += reward
        
        if self.parent is not None:
            self.parent.backup(reward)
    
    def rewardFunction(self, state, initScore):
        features = self.getFeatures(state, initScore)
        weights = self.getWeights()

        return features * weights
    
    def getFeatures(self, state, initScore):
        features = util.Counter()
        enemies = [state.getAgentState(opp) for opp in self.agent.getOpponents(state)]
        invaders = [opp for opp in enemies if opp.isPacman and opp.getPosition() != None]
        defenders = [opp for opp in enemies if (not opp.isPacman) and opp.getPosition() != None]
        food = self.agent.getFood(state).asList()
        oppFood = self.agent.getFoodYouAreDefending(state).asList()


        # Get the position and carrying status of the agent
        agent_pos = state.getAgentPosition(self.agentIndex)
        #is_pacman = state.getAgentState(self.agentIndex).isPacman
        features['carry'] = state.getAgentState(self.agentIndex).numCarrying
        features['score'] = self.agent.getScore(state) - initScore
        if not self.parent is None:
            if not self.parent.action is None:
                if self.action == Directions.REVERSE[self.parent.action] and self.agentIndex == self.index: features['reverse'] = 1
        if self.action == Directions.STOP and self.agentIndex == self.index: features['stop'] = 1

        # Get the number of opponents scared and caught
        #opponents_scared = sum([1 for opp in enemies if state.getAgentState(opp).scaredTimer > 0])
        #opponents_caught = sum([1 for opp in enemies if state.getAgentState(opp).isPacman and state.getAgentState(opp).numCarrying == 0])
        features['oppCarry'] = sum([opp.numCarrying for opp in enemies])

        if len(food) > 0:
            features['distFood'] = min([self.agent.getMazeDistance(agent_pos, food_pos) for food_pos in food])

        if len(oppFood) > 0:
            features['distOppFood'] = min([self.agent.getMazeDistance(agent_pos, food_pos) for food_pos in oppFood])

        if len(invaders) > 0:
            features['distInvaders'] = min([self.agent.getMazeDistance(agent_pos, opp.getPosition()) for opp in invaders])

        if len(defenders) > 0:
            features['distDefenders'] = min([self.agent.getMazeDistance(agent_pos, opp.getPosition()) for opp in defenders])

        return features
    
    def getWeights(self):
        if self.mode == "attack":
            return {'score': 1000.0, 'distFood': -1.0, 'distOppFood': 0.0, 'distInvaders': -0.2, 'distDefenders': 0.8, 'carry': 10.0, 'oppCarry': 0.0, 'stop': -2.0, 'reverse': -1.0}
        
        elif self.mode == "defense":
            return {'score': 0.0, 'distFood': 0.0, 'distOppFood': -1.0, 'distInvaders': -10.0, 'distDefenders': 0.0, 'carry': 0.0, 'oppCarry': -10.0, 'stop': -2.0, 'reverse': -1.0}

    def printRewards(self, initScore):
        print(" ")
        print("Agent Index: " + str(self.agentIndex) + " | Agent position: " + str(self.state.getAgentPosition(self.agentIndex)) + " | visits: " + str(self.visits) + " | Carrying: " + str(self.state.getAgentState(self.agentIndex).numCarrying))
        print("---- ACTIONS: REWARDS / VISITS / AVERAGE ----")
        self.printTree("- ", initScore)

    def printTree(self, string, initScore):
        for child in self.children:
            rewards = child.evaluateTree(initScore)
            print(string + child.action + ": " + str(rewards) + " / " + str(child.visits) + " / " + str(rewards/self.visits))
            #child.printTree("   " + string)

    def evaluateTree(self, initScore):
        if self.isFullyExpanded():
            if self.myTurn:
                score = max([child.evaluateTree(initScore) for child in self.children])
            else:
                score = min([child.evaluateTree(initScore) for child in self.children])
        else:
            return self.rewardFunction(self.state, initScore)

        return score

    def minmaxAction(self, initScore):
        maxScore = -float("inf")
        bestChild = None
        for child in self.children:
            score = child.evaluateTree(initScore)
            if score > maxScore:
                bestChild = child
                maxScore = score
        return bestChild.action


 

class MCTSPacman(CaptureAgent):
    def __init__(self, index, mode="attack", depth=1, timeForComputing=0.2):
        CaptureAgent.__init__(self, index)
        self.timeForComputing = timeForComputing
        self.depth = depth
        self.mode = mode

    def registerInitialState(self, state):
        CaptureAgent.registerInitialState(self, state)
        self.start = state.getAgentPosition(self.index)
        self.prevAction = None
        
    def chooseAction(self, initState):
        # Set the start state
        state = initState.deepCopy()
        print(self.getFood(state).asList())
        # Create the root node
        rootNode = Node(state, self.index, self.index, depth=self.depth, action=self.prevAction, agent=self, mode=self.mode)
        initScore = self.getScore(state)
        # Start the MCTS algorithm
        i=0
        startTime = time.time()
        while time.time() - startTime < self.timeForComputing:
            # Select - Tree Policy
            node = rootNode.treePolicy()

            reward = node.defaultPolicy()

            # Backup
            node.backup(reward)


        #self.prevAction = rootNode.bestChild(c=0).action
        #rootNode.printRewards(initScore)
        # Return the best action
        return rootNode.minmaxAction(initScore)