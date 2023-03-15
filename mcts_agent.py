import random
import math
from captureAgents import CaptureAgent
import random
import time

class Node:
    def __init__(self, state, agentIndex, parent=None, action=None):
        self.state = state
        self.agentIndex = agentIndex
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.totalReward = 0
        
    def isLeaf(self):
        return len(self.children) == 0
    
    def isTerminal(self):
        return self.state.isOver()
    
    def selectChild(self):
        UCT_CONSTANT = 1.0
        bestScore = float('-inf')
        bestChild = None
        
        for child in self.children:
            exploitation = child.totalReward / child.visits
            exploration = UCT_CONSTANT * (math.log(self.visits) / child.visits) ** 0.5
            score = exploitation + exploration
            
            if score > bestScore:
                bestScore = score
                bestChild = child
        
        return bestChild
    
    def expand(self):
        actions = self.state.getLegalActions(self.agentIndex)
        action = random.choice(actions)
        nextState = self.state.generateSuccessor(self.agentIndex, action)
        child = Node(nextState, self.agentIndex, self, action)
        self.children.append(child)
        
        return child
    
    def simulate(self):
        state = self.state.deepCopy()
        agentIndex = self.agentIndex
        while not state.isOver():
            actions = state.getLegalActions(agentIndex)
            action = random.choice(actions)
            state = state.generateSuccessor(agentIndex, action)
            
        return state.getScore()
    
    def backpropagate(self, reward):
        self.visits += 1
        self.totalReward += reward
        
        if self.parent is not None:
            self.parent.backpropagate(reward)
        
    def bestAction(self):
        bestChild = max(self.children, key=lambda child: child.visits)
        return bestChild.action

class MCTSAgent(CaptureAgent):
    def __init__(self, index, timeForComputing=.1):
        CaptureAgent.__init__(self, index)
        self.timeForComputing = timeForComputing
        
    def getAction(self, gameState):
        # Set the start state
        state = gameState.deepCopy()
        
        # Create the root node
        rootNode = Node(state, self.index)
        
        # Start the MCTS algorithm
        startTime = time.time()
        while time.time() - startTime < self.timeForComputing:
            node = rootNode
            
            # Select
            while not node.isLeaf():
                node = node.selectChild()
            
            # Expand
            if not node.isTerminal():
                node = node.expand()
                
            # Simulate
            result = node.simulate()
            
            # Backpropagate
            node.backpropagate(result)
            
        # Return the best action
        return rootNode.bestAction()
