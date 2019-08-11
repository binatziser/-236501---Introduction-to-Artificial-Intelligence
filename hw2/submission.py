import random, util
from game import Agent
from ghostAgents import DirectionalGhost

MIN_DIST_FROM_GHOST = 1.4					# This is the minimal distance from ghosts we allow the pacman to be.
# We wanted the highest score - so its only logical to use the score in the heuristics.
SCORE_MULTI = 4								# Multiplies the game score.
GAME_LOSE_BONUS = -200						# Additional bonus given when we lose.
# These parameters only work when there isn't a lot of food on the board (SMALL_AMOUNT_ON_FOOD_*)
# We calculate small amount of food as: foodGrid.width * foodGrid.height / 10
# We added these weights because we wanted the pacman to be more Food oriented when the food was low, and because the small food amount made it
# harder for the pacman to choose the right direction to the food.
SMALL_AMOUNT_OF_FOOD_MUL = 1				# Multiplier for the amount of food regarded as small amount of food.
SMALL_AMOUNT_OF_FOOD_MANHATTAN_MUL = 1		# Multiplies the average menhatten distance from the food.
SMALL_AMOUNT_OF_FOOD_CLOSE_FOOD_MUL = 0.93	# Multiplies the weight for the closest food.
SMALL_AMOUNT_OF_FOOD_SAME_DIR_MUL = 1.6		# Multiplies the weight for pacman moving in the same direction
SMALL_AMOUNT_OF_FOOD_CLOSE_GHOST_MUL = 0.8	# Multiplies the weight for the ghosts that are close to us.
SMALL_AMOUNT_OF_FOOD_MEDIUM_GHOST_MUL = 0.4	# Multiplies the weight for the ghosts that are in medium distance from us.
SMALL_AMOUNT_OF_FOOD_FAR_FOOD_MUL = 0.525	# Multiplies the weight for the furthest food.
SMALL_AMOUNT_OF_FOOD_FAR_GHOST_MUL = 0.2	# Multiplies the weight for the ghosts that are far from us.
SMALL_AMOUNT_OF_FOOD_SCARED_DIST_MUL = 0.25	# Multiplies the weight for the ghosts that are scared.
SMALL_AMOUNT_OF_FOOD_CAPSULE_MUL = 0.9		# Multiplies the weight for taking a capsule.
# In some cases we will want to stop pacman from eating capsules - like in the case that there are scared ghosts and eating the capsule wont change anything.
SCARED_CAPSULE_MUL = -0.5					# Multiplies the bonus we get for eating a capsule while there are still scared ghosts.
CAPSULE_BONUS = -260						# A bonus we get for every capsule in the layout (so eating a capsule will result in a higher score).
# Bonuses for ghosts, we also considered far ghosts to add another level to the heuristics that will help pacman when he is stuck.
# We wanted to differ between the distances of the ghosts and give them weights according to how dangerous they are to us.
GHOST_SCARED_DIST_MUL = -3.6				# Multiplies the distance regarded when trying to ass
GHOST_EAT_BONUS = 150						# Bonus for eating a ghost.
GHOST_DIST_BONUS = 0.4      				# Multiplies the distance from the ghosts.
GHOST_FAR_BONUS = -1         				# Multiplies the number of far ghosts.
GHOST_MEDIUM_BONUS = 0.9					# Multiplies the number of medium ghosts.
GHOST_BAD_BONUS = -480						# Multiplies the number of ghosts closer than MIN_DIST_FROM_GHOST

MIN_DIST_MULTIPLIER = 2.7     				# Multiplier for MIN_DIST_FROM_GHOST - distance for saying whats a medium ghost is
											# Notice: all other ghosts will be considered far ghosts.
DIST_IF_SCARED_MUL = 1.38					# Multiplier for MIN_DIST_FROM_GHOST considering what in an eatable ghost.
# These params are to make pacman be more food oriented and to make him prefer closer food than far food but still not try and find the best weights.
FOOD_FAR_BONUS = -0.5						# Multiplies the distance of the furthests food.
FOOD_CLOSE_BONUS = -5						# Multiplies the distance of the closest food.
NUM_FOOD_BONUS = -10         				# Multiplies the total number of food - to make pacman be more food oriented.
NO_FOOD_BONUS = 0          					# To help pacman not leave 1 food left alone, considers close food around him - WE FAILED TO MAKE THIS WORK @$#%@%$!#
MANHATTAN_ENABLE = -1.1       				# Multiplies the average menhatten distance from the food.

SAME_DIR_BONUS = 2.25						# A bonus given if pacman is continueing in the same vector as last turn.

#     ********* Reflex agent- sections a and b *********

class OriginalReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
        ------------------------------------------------------------------------------
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current GameState (pacman.py) and the proposed action
        and returns a number, where higher numbers are better.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        return scoreEvaluationFunction(successorGameState)


#     ********* Evaluation functions *********

def scoreEvaluationFunction(gameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.
    """
    return gameState.getScore()

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
        ------------------------------------------------------------------------------
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        pacDir = gameState.getPacmanState().getDirection()
        i = 0
        for move in legalMoves:
            if move is 'North' and pacDir is 'North' and 'South' in legalMoves:
                scores[i] += SAME_DIR_BONUS * SMALL_AMOUNT_OF_FOOD_SAME_DIR_MUL
            if move is 'South' and pacDir is 'South' and 'North' in legalMoves:
                scores[i] += SAME_DIR_BONUS * SMALL_AMOUNT_OF_FOOD_SAME_DIR_MUL
            if move is 'East' and pacDir is 'East' and 'West' in legalMoves:
                scores[i] += SAME_DIR_BONUS * SMALL_AMOUNT_OF_FOOD_SAME_DIR_MUL
            if move is 'West' and pacDir is 'West' and 'East' in legalMoves:
                scores[i] += SAME_DIR_BONUS * SMALL_AMOUNT_OF_FOOD_SAME_DIR_MUL
            i += 1

        # print("legalMoves")
        # print(legalMoves)
        # print("score")
        # print(scores)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # print(bestScore)
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current GameState (pacman.py) and the proposed action
        and returns a number, where higher numbers are better.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        return betterEvaluationFunction(successorGameState)


#     ********* Evaluation functions *********

def scoreEvaluationFunction(gameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.
    """
    return gameState.getScore()


######################################################################################
# b: implementing a better heuristic function
def betterEvaluationFunction(gameState):
    """

    The betterEvaluationFunction takes in a GameState (pacman.py) and should return a number, where higher numbers are better.

    A GameState specifies the full game state, including the food, capsules, agent configurations and more.
    Following are a few of the helper methods that you can use to query a GameState object to gather information about
    the present state of Pac-Man, the ghosts and the maze:

    gameState.getLegalActions():
    gameState.getPacmanState():
    gameState.getGhostStates():
    gameState.getNumAgents():
    gameState.getScore():
    The GameState class is defined in pacman.py and you might want to look into that for other helper methods.
    """
    if gameState.isLose():
        return -20000

    pacState = gameState.getPacmanState()
    pacPos = gameState.getPacmanPosition()
    pacDir = gameState.getPacmanState().getDirection()
    pacPos_x = pacPos[0]
    pacPos_y = pacPos[1]
    legalActions = gameState.getLegalActions()

    numAgents = gameState.getNumAgents()

    # Capsules
    capsulesNum = len(gameState.getCapsules())

    # Food
    foodGrid = gameState.getFood()
    foodAroundPac = False
    if pacPos_x != 0:
        foodAroundPac = foodAroundPac or (foodGrid[pacPos_x - 1][pacPos_y] and 'West' in legalActions)
    if pacPos_y != 0:
        foodAroundPac = foodAroundPac or (foodGrid[pacPos_x][pacPos_y - 1] and 'South' in legalActions)
    if pacPos_x != foodGrid.width-1:
        foodAroundPac = foodAroundPac or (foodGrid[pacPos_x + 1][pacPos_y] and 'East' in legalActions)
    if pacPos_y != foodGrid.height-1:
        foodAroundPac = foodAroundPac or (foodGrid[pacPos_x][pacPos_y + 1] and 'North' in legalActions)

    noFoodAroundPac = 1
    if foodAroundPac:
        noFoodAroundPac = 0

    SMALL_AMOUNT_OF_FOOD = foodGrid.width * foodGrid.height / 10 * SMALL_AMOUNT_OF_FOOD_MUL

    notALotOfFood = False
    if foodGrid.count() < SMALL_AMOUNT_OF_FOOD:
        notALotOfFood = True

    closestFood = float('inf')
    farthestFood = -1
    total_dist = 0

    for x in range(foodGrid.width):             # loop for checking food
        for y in range(foodGrid.height):
            if foodGrid[x][y]:
                dist = util.manhattanDistance((x, y), pacPos)
                total_dist += dist
                if dist < closestFood:
                    closestFood = dist
                if dist > farthestFood:
                    farthestFood = dist
    #print("far")
    #print(farthestFood)
    #print("close")
    #print(closestFood)

    if foodGrid.count() != 0:
        average_dist = total_dist/foodGrid.count()
    else:
        average_dist = 0

    # TODO: added for openClassic shit
    # if foodGrid.count() < 5:
    #     farthestFood = 0
    #     average_dist = 0

    if notALotOfFood:
        average_dist *= SMALL_AMOUNT_OF_FOOD_MANHATTAN_MUL
        closestFood *= SMALL_AMOUNT_OF_FOOD_CLOSE_FOOD_MUL
        farthestFood *= SMALL_AMOUNT_OF_FOOD_FAR_FOOD_MUL
        # Ghosts
    ghostsState = gameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostsState]

    ghostPositions = gameState.getGhostPositions()
    closeGhostsNum = 0
    mediumGhostsNum = 0
    farGhostsNum = 0
    farGhostDist = 0
    eatableGhosts = 0
    scaredGhosts = 0
    scaredFarGhosts = 0
    scaredGhostsDist = 0

    isThereAScaredGhost = False

    for gPos, scaredTime in zip(ghostPositions, scaredTimes):  # loop for checking ghosts
        dist = util.manhattanDistance(gPos, pacPos)
        if scaredTime > 0:
            isThereAScaredGhost = True
            scaredGhosts += 1
            scaredGhostsDist += dist
            if dist <= (DIST_IF_SCARED_MUL * MIN_DIST_FROM_GHOST):  # TODO: maybe not good
                eatableGhosts += 1
            elif dist > 2 * MIN_DIST_MULTIPLIER * MIN_DIST_FROM_GHOST:
                scaredFarGhosts += 1
        elif dist <= MIN_DIST_FROM_GHOST:
            closeGhostsNum += 1  # near us but not eatable
        elif dist < dist <= MIN_DIST_MULTIPLIER * MIN_DIST_FROM_GHOST:
            mediumGhostsNum += 1
        else:
            farGhostsNum += 1
            farGhostDist += dist

    farGhostScore = 0
    if not isThereAScaredGhost:
        farGhostScore = farGhostsNum * GHOST_FAR_BONUS + farGhostDist * GHOST_DIST_BONUS
    # else:
    # print('scared ghosts')

    # Score
    score = gameState.getScore() * SCORE_MULTI + int(gameState.isLose()) * GAME_LOSE_BONUS
    mediumGhostScore = mediumGhostsNum * GHOST_MEDIUM_BONUS
    capsuleScore = capsulesNum * CAPSULE_BONUS
    if notALotOfFood:
        farGhostScore *= SMALL_AMOUNT_OF_FOOD_FAR_GHOST_MUL
        mediumGhostScore *= SMALL_AMOUNT_OF_FOOD_MEDIUM_GHOST_MUL
        closeGhostsNum *= SMALL_AMOUNT_OF_FOOD_CLOSE_GHOST_MUL
        capsuleScore *= SMALL_AMOUNT_OF_FOOD_CAPSULE_MUL
        scaredGhostsDist *= SMALL_AMOUNT_OF_FOOD_SCARED_DIST_MUL
    elif scaredFarGhosts >= scaredGhosts / 2 > 0 and mediumGhostsNum + closeGhostsNum == 0 and \
            closeGhostsNum + mediumGhostsNum + farGhostsNum < scaredGhosts:
        capsuleScore *= SCARED_CAPSULE_MUL
    if foodGrid.count() == 1 and closestFood < 2:
        closestFood *= 1
    ghostScore = closeGhostsNum ** 2 * GHOST_BAD_BONUS + eatableGhosts * GHOST_EAT_BONUS + farGhostScore + \
                 mediumGhostScore + scaredGhostsDist * GHOST_SCARED_DIST_MUL
    foodScore = closestFood * FOOD_CLOSE_BONUS + farthestFood * FOOD_FAR_BONUS + \
                average_dist * MANHATTAN_ENABLE + foodGrid.count() * NUM_FOOD_BONUS + \
                noFoodAroundPac * NO_FOOD_BONUS

    heuristicsScore = score + ghostScore + foodScore + capsuleScore

    if gameState.isWin():
        return 1000000 + score
    return heuristicsScore
    #     ********* MultiAgent Search Agents- sections c,d,e,f*********

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxAgent, AlphaBetaAgent & both ExpectimaxAgents.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


######################################################################################
# c: implementing minimax

# NATI_SUPER_DUPER_CRAZY

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent
    """
    def getAction(self, gameState):
        numGhosts = gameState.getNumAgents() - 1

        def maxLevel(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            legalActions = gameState.getLegalActions(0)
            successors = [gameState.generateSuccessor(0, action) for action in legalActions]
            successorsScore = [minLevel(suc, depth, 1) for suc in successors]
            bestScore = max(successorsScore)
            return bestScore if bestScore > -(float("inf")) else -(float("inf"))

        def minLevel(gameState, depth, agentindex):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            legalActions = gameState.getLegalActions(agentindex)
            successors = [gameState.generateSuccessor(agentindex, action) for action in legalActions]
            if agentindex == numGhosts:
                successorsScore = [maxLevel(suc, depth - 1) for suc in successors]
            else:
                successorsScore = [minLevel(suc, depth, agentindex + 1) for suc in successors]
            bestScore = min(successorsScore)
            return bestScore if bestScore < float("inf") else float("inf")

        legalActions = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(0, action) for action in legalActions]
        successorsScore = [minLevel(suc, self.depth, 1) for suc in successors]
        bestScore = max(successorsScore)
        bestIndices = [index for index in range(len(successorsScore)) if successorsScore[index] == bestScore]
        return legalActions[bestIndices[0]] if bestScore > -(float("inf")) else "STOP"

######################################################################################
# d: implementing alpha-beta

# NATI_SUPER_DUPER_CRAZY
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning
  """
    def getAction(self, gameState):
        numGhosts = gameState.getNumAgents() - 1

        def maxLevel(gameState, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            v = -(float("inf"))
            legalActions = gameState.getLegalActions(0)
            for action in legalActions:
                v = max(v, minLevel(gameState.generateSuccessor(0, action), depth, 1, alpha, beta))
                alpha = max(v, alpha)
                if v >= beta:
                    return float("inf")
            return v

        def minLevel(gameState, depth, agentindex, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            v = float("inf")
            legalActions = gameState.getLegalActions(agentindex)
            if agentindex == numGhosts:
                for action in legalActions:
                    v = min(v, maxLevel(gameState.generateSuccessor(agentindex, action), depth - 1, alpha, beta))
                    beta = min(v, beta)
                    if v <= alpha:
                        return -(float("inf"))
                return v
            else:
                for action in legalActions:
                    v = min(v, minLevel(gameState.generateSuccessor(agentindex, action), depth, agentindex + 1, alpha,
                                        beta))
                    beta = min(v, beta)
                    if v <= alpha:
                        return -(float("inf"))
                return v

        legalActions = gameState.getLegalActions()
        alpha = -(float("inf"))
        maxi = -(float("inf"))
        bestAction = "STOP"
        for action in legalActions:
            score = minLevel(gameState.generatePacmanSuccessor(action), self.depth, 1, alpha, (float("inf")))
            if score > maxi:
                alpha = score
                maxi = score
                bestAction = action

        return bestAction

######################################################################################
# e: implementing random expectimax

class RandomExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent
    """
    def __init__(self, evalFn='betterEvaluationFunction', depth='0'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their legal moves.
        """

        numGhosts = gameState.getNumAgents() - 1

        def maxLevel(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            legalActions = gameState.getLegalActions(0)
            successors = [gameState.generateSuccessor(0, action) for action in legalActions]
            successorsScore = [expectiLevel(suc, depth, 1) for suc in successors]
            bestScore = max(successorsScore)
            return bestScore if bestScore > -(float("inf")) else depth (float("inf"))

        def expectiLevel(gameState, depth, agentindex):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            legalActions = gameState.getLegalActions(agentindex)
            probability = 1 / len(legalActions)
            successors = [gameState.generateSuccessor(agentindex, action) for action in legalActions]
            if agentindex == numGhosts:
                successorsScore = probability * sum([maxLevel(suc, depth - 1) for suc in successors])
            else:
                successorsScore = probability * sum([expectiLevel(suc, depth, agentindex + 1) for suc in successors])
            # bestScore = min(successorsScore)
            return successorsScore if successorsScore < float("inf") else float("inf")

        legalActions = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(0, action) for action in legalActions]
        successorsScore = [expectiLevel(suc, self.depth, 1) for suc in successors]
        bestScore = max(successorsScore)
        bestIndices = [index for index in range(len(successorsScore)) if successorsScore[index] == bestScore]
        return legalActions[bestIndices[0]] if bestScore > -(float("inf")) else "STOP"


######################################################################################
# f: implementing directional expectimax

class DirectionalExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as using the DirectionalGhost distribution to choose from their legal moves.
        """

        numGhosts = gameState.getNumAgents() - 1

        def maxLevel(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            legalActions = gameState.getLegalActions(0)
            successors = [gameState.generateSuccessor(0, action) for action in legalActions]
            successorsScore = [expectiLevel(suc, depth, 1) for suc in successors]
            bestScore = max(successorsScore)
            return bestScore if bestScore > -(float("inf")) else -(float("inf"))

        def expectiLevel(gameState, depth, agentindex):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            legalActions = gameState.getLegalActions(agentindex)
            ghostState = DirectionalGhost(agentindex, prob_attack=0.8, prob_scaredFlee=0.8 )
            p = DirectionalGhost.getDistribution(ghostState, gameState)
            successors = [gameState.generateSuccessor(agentindex, action) for action in legalActions]
            if agentindex == numGhosts:
                successorsScore = sum([p[action] * maxLevel(suc, depth - 1) for (suc, action) in zip(successors, legalActions)])
            else:
                successorsScore = sum([p[action] * expectiLevel(suc, depth, agentindex + 1) for (suc, action) in zip(successors, legalActions)])
            # bestScore = min(successorsScore)
            return successorsScore if successorsScore < float("inf") else float("inf")

        legalActions = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(0, action) for action in legalActions]
        successorsScore = [expectiLevel(suc, self.depth, 1) for suc in successors]
        bestScore = max(successorsScore)
        bestIndices = [index for index in range(len(successorsScore)) if successorsScore[index] == bestScore]
        return legalActions[bestIndices[0]] if bestScore > -(float("inf")) else "STOP"


######################################################################################
# I: implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent):
    """
      Your competition agent
    """

    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their legal moves.
        """

        numGhosts = gameState.getNumAgents() - 1

        def maxLevel(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            legalActions = gameState.getLegalActions(0)
            successors = [gameState.generateSuccessor(0, action) for action in legalActions]
            successorsScore = [expectiLevel(suc, depth, 1) for suc in successors]
            bestScore = max(successorsScore)
            return bestScore if bestScore > -(float("inf")) else depth(float("inf"))

        def expectiLevel(gameState, depth, agentindex):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            legalActions = gameState.getLegalActions(agentindex)
            probability = 1 / len(legalActions)
            successors = [gameState.generateSuccessor(agentindex, action) for action in legalActions]
            if agentindex == numGhosts:
                successorsScore = probability * sum([maxLevel(suc, depth - 1) for suc in successors])
            else:
                successorsScore = probability * sum([expectiLevel(suc, depth, agentindex + 1) for suc in successors])
            # bestScore = min(successorsScore)
            return successorsScore if successorsScore < float("inf") else float("inf")

        legalActions = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(0, action) for action in legalActions]
        successorsScore = [expectiLevel(suc, self.depth, 1) for suc in successors]
        bestScore = max(successorsScore)
        bestIndices = [index for index in range(len(successorsScore)) if successorsScore[index] == bestScore]
        return legalActions[bestIndices[0]] if bestScore > -(float("inf")) else "STOP"
