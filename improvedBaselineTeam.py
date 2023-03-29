# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class improvedReflexAgent(CaptureAgent):
  """
  A base class for the new Improved Reglex Agent
  that will introduce the following approaches:
  - Secure food on batches
  - Avoid Enemies
  - Try to eat enemies
  """
  movesAttack = 0
  movesDefense = 0

  def registerInitialState(self, gameState):
    """
    Initial setup of the agent
    """
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions that are most suited for the situation.
    """
    # Possible actions
    actions = gameState.getLegalActions(self.index)

    # Info of the game
    team = self.getTeam(gameState)
    enemyIndex = self.getOpponents(gameState)
    bellyCapacity = 1

    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    
    # When atacking
    if team[0] == self.index:
      # How many food we have eaten
      eatenPellets = gameState.getAgentState(self.index).numCarrying

      # When one is eaten check
      if eatenPellets >= bellyCapacity:
        bestDist = 9999
        enemDist = 9999
        for action in actions:
          # Get next action
          successor = self.getSuccessor(gameState, action)

          # positions in next action
          pos2 = successor.getAgentPosition(self.index)
          enPos = [successor.getAgentPosition(i) for i in enemyIndex]
          
          # Distance in next action
          dist = self.getMazeDistance(self.start,pos2)
          dist_enemy = self.getMazeDistance(pos2, min(enPos))

          # Enemy close to escape rute
          if dist_enemy < enemDist:
            bestEscape = action
            enemDist = dist_enemy
          
          # If no enemy close
          if dist < bestDist:
            bestAction = action
            bestDist = dist

        # Desition
        if dist_enemy < 3:
          return bestEscape
        else:
          return bestAction

    # Not eaten yet, go to close food
    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Linear combination of features and feature weights.
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state, mostly depends on
    the type of agent, if offensive or defensive.
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self):
    """
    The wheigts.
    Returns a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(improvedReflexAgent):
  """
  Offensive agent, goes for the food and secures it
  """
  def getFeatures(self, gameState, action):
    # Initialize Features
    features = util.Counter()

    # Game information
    initScore = copy.deepcopy(self.getScore(gameState))
    agent_pos = gameState.getAgentPosition(self.index)

    # Food information
    foodList = self.getFood(gameState).asList()
    oppFoodList = self.getFoodYouAreDefending(gameState).asList()
    
    # Enenmy information
    enemies = [gameState.getAgentState(opp) for opp in self.getOpponents(gameState)]
    invaders = [opp for opp in enemies if opp.isPacman and opp.getPosition() != None]
    defenders = [opp for opp in enemies if (not opp.isPacman) and opp.getPosition() != None]

    # Add features!
    if len(foodList) > 0:
      # Get the position and carrying status of the agent
      features['carry'] = gameState.getAgentState(self.index).numCarrying
      features['score'] = self.getScore(gameState) - initScore

      # Get the number of opponents scared and caught
      features['oppCarry'] = sum([opp.numCarrying for opp in enemies])
      features['distFood'] = min([self.getMazeDistance(agent_pos, food_pos) for food_pos in foodList])

      if len(oppFoodList) > 0:
        features['distOppFood'] = min([self.getMazeDistance(agent_pos, food_pos) for food_pos in oppFoodList])

      if len(invaders) > 0:
        features['distInvaders'] = min([self.getMazeDistance(agent_pos, opp.getPosition()) for opp in invaders])

      if len(defenders) > 0:
        features['distDefenders'] = min([self.getMazeDistance(agent_pos, opp.getPosition()) for opp in defenders])

      if action == Directions.STOP: features['stop'] = 1
      
      rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
      if action == rev: features['reverse'] = 1

    return features

  def getWeights(self):
    return {'score': 1000.0, 'distFood': -1.0, 'distOppFood': 0.0, 'distInvaders': -0.5, 'distDefenders': 0.5, 'carry': 10.0, 'oppCarry': 0.0, 'stop': -2.0, 'reverse': -1.0}

class DefensiveReflexAgent(improvedReflexAgent):
  """
  Defensive agent, tries to eat any packman that enters the team base
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    # print(enemies[0])
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1
    # print(features)
    return features

  def getWeights(self):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
