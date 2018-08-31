from Simulator import Simulation
import numpy as np

class Engine:

    def __init__(self, nbSimulation, robotShape, goalShape, boardSize):

        self.boardSize = boardSize
        self.robotShape = robotShape
        self.goalShape = goalShape
        self.robotShape = robotShape
        self.nbSimulation = nbSimulation
        self.simulationList = []
        self.allRobotPos = np.zeros((self.nbSimulation,2),dtype=np.int32)
        self.allGoalPos = np.zeros((self.nbSimulation,2),dtype=np.int32)
        self.actionMove = self.createActionMoveAssociation()
        self.quadrant = self.createPossiblePosition()
        self.createSimulation()

    def createSimulation(self):

        self.simulationList = []
        self.allRobotPos = np.zeros((self.nbSimulation,2),dtype=np.int32)
        self.allGoalPos = np.zeros((self.nbSimulation,2),dtype=np.int32)

        for i in range(self.nbSimulation):

            nb1 = np.random.randint(0,high=(4))
            nb2 = np.random.randint(0,high=(4))
            while (nb2 == nb1):
                nb2 = np.random.randint(0,high=(4))

            quadrant1 = self.quadrant[nb1]
            quadrant2 = self.quadrant[nb2]

            xRobotPos = np.random.randint(quadrant1[0][0],high=(quadrant1[0][1] - self.robotShape[0]))
            yRobotPos = np.random.randint(quadrant1[1][0],high=(quadrant1[1][1] - self.robotShape[1]))
            xGoalPos = np.random.randint(quadrant2[0][0],high=(quadrant2[0][1] - self.goalShape[0]))
            yGoalPos = np.random.randint(quadrant2[1][0],high=(quadrant2[1][1] - self.goalShape[1]))

            self.allRobotPos[i,0] = xRobotPos
            self.allRobotPos[i,1] = yRobotPos
            self.allGoalPos[i,0] = xGoalPos
            self.allGoalPos[i,1] = yGoalPos

            self.simulationList += [Simulation((xRobotPos,yRobotPos), self.robotShape, (xGoalPos,yGoalPos), self.goalShape, self.boardSize)]

    def update(self, robotMoveList):

        for i in range(self.nbSimulation):
            currentMove = self.actionMove[robotMoveList[i]]
            currentMove = self.checkInLimit(self.allRobotPos[i],currentMove)
            self.allRobotPos[i] = self.allRobotPos[i] + currentMove
            self.simulationList[i].update(currentMove)

    def getAllRobotPos(self):

        return np.copy(self.allRobotPos)

    def checkInLimit(self, pos, move):

        if (not (pos[0] + move[0] < self.boardSize - self.robotShape[0])):
            move[0] = 0

        elif (not (0 < pos[0] + move[0])):
            move[0] = 0

        if (not (pos[1] + move[1] < self.boardSize-self.robotShape[1])):
            move[1] = 0

        elif (not (0 < pos[1] + move[1])):
            move[1] = 0

        return move

    def getAllGoalPos(self):

        return np.copy(self.allGoalPos)

    def drawAllBoard(self):

        boardList = np.zeros((self.nbSimulation,self.boardSize,self.boardSize))

        for i in range(self.nbSimulation):
            boardList[i] = self.simulationList[i].drawBoard()

        return boardList

    def createAllPenaltyBoard(self):

        boardList = []

        for simulation in self.simulationList:

            boardList += [simulation.createPenaltyBoard()]

        return boardList

    def calculateStepReward(self, oldRobotPos, newRobotPos):

        oldDist = self.getDist(oldRobotPos)
        newDist = self.getDist(newRobotPos)
        penaltyList = self.calculateObstaclePenalty(newRobotPos)
        penaltyList = np.minimum(penaltyList, 1)

        diff = oldDist - newDist
        norm = diff / 14
        reward = np.maximum(norm, 0)
        reward = reward - penaltyList

        return reward

    def calculateFinalReward(self, robotPos):

        reward = self.getDist(robotPos)
        reward[reward <= 25] = 1
        #reward[(reward <= 75) & (reward > 25)] = 2
        reward[reward > 25] = 0

        return reward

    def getDist(self,robotPos):

        diff = self.allGoalPos - robotPos
        power = diff**2
        dist = np.sqrt(np.sum(power,axis=1))

        return dist

    def calculateObstaclePenalty(self, newRobotPos):

        penaltyList = np.zeros((self.nbSimulation))

        for i in range(self.nbSimulation):

            penaltyBoard = self.simulationList[i].createPenaltyBoard()
            allPenalty = penaltyBoard[newRobotPos[i,0]:newRobotPos[i,0] + self.robotShape[0], newRobotPos[i,1]:newRobotPos[i,1] + self.robotShape[1]]
            penaltyList[i] = np.sum(allPenalty)

        return penaltyList

    def createActionMoveAssociation(self):

        moveDist = 10
        moveDistMinus = moveDist * -1

        actionMoveDict = {
            0:np.array([moveDist,0]),
            1:np.array([0,moveDist]),
            2:np.array([moveDistMinus,0]),
            3:np.array([0,moveDistMinus]),
            4:np.array([moveDist,moveDist]),
            5:np.array([moveDistMinus,moveDistMinus]),
            6:np.array([moveDist,moveDistMinus]),
            7:np.array([moveDistMinus,moveDist])
        }

        return actionMoveDict

    def createPossiblePosition(self):

        half = int(self.boardSize / 2)
        halfMinus = half - 25
        halfPlus = half + 25

        leftTop = (0,halfMinus)
        rigthBottom = (halfPlus,self.boardSize-1)

        possiblePos = {
            0:(leftTop,leftTop),
            1:(leftTop,rigthBottom),
            2:(rigthBottom,rigthBottom),
            3:(rigthBottom,leftTop)
        }

        return possiblePos