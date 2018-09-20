import numpy as np

class Obstacle:

    def __init__(self, obstacleShape, obstaclePos, boardSize):

        self.obstacleShape = obstacleShape
        self.obstaclePos = obstaclePos
        self.boardSize = boardSize

    def drawOnBoard(self, board):

        board[self.obstaclePos[0]:self.obstaclePos[0]+self.obstacleShape[0], self.obstaclePos[1]:self.obstaclePos[1]+self.obstacleShape[1]] = 1

    def putSubPenaltyOnBoard(self, board):

        xStart = self.obstaclePos[0] - 5
        if (xStart < 0):
            xStart = self.obstaclePos[0]

        yStart = self.obstaclePos[1] - 5
        if (yStart < 0):
            yStart = self.obstaclePos[1]

        xEnd = self.obstaclePos[0] + self.obstacleShape[0] + 5
        if (self.boardSize < xEnd):
            xEnd = self.obstaclePos[0] + self.obstacleShape[0]

        yEnd = self.obstaclePos[1] + self.obstacleShape[1] + 5
        if (self.boardSize < yEnd):
            yEnd = self.obstaclePos[1] + self.obstacleShape[1]
        board[xStart:xEnd, yStart:yEnd] = 0.01

    def putPenaltyOnBoard(self, board):

        board[self.obstaclePos[0]:self.obstaclePos[0]+self.obstacleShape[0], self.obstaclePos[1]:self.obstaclePos[1]+self.obstacleShape[1]] = 1

class Simulation:

    def __init__(self, robotPos, robotShape, goalPos, goalShape, boardSize):

        self.boardSize = boardSize
        self.robotPos = robotPos
        self.robotShape = robotShape
        self.goalPos = goalPos
        self.goalShape = goalShape
        self.nbObstacle = np.random.randint(1, high=21)
        self.obstacleMaxSize = 51
        self.obstacleList = []
        self.createObstacles()

    def createObstacles(self):

        self.obstacleList = []

        for i in range(self.nbObstacle):

            height = np.random.randint(5, high=self.obstacleMaxSize)
            width = np.random.randint(5, high=self.obstacleMaxSize)

            xPos = np.random.randint(0,high=self.boardSize-width)
            while (self.collideWithRobotOrGoal(xPos,width,0)):
                xPos = np.random.randint(0,high=self.boardSize-width)

            yPos = np.random.randint(0,high=self.boardSize-height)
            while (self.collideWithRobotOrGoal(yPos,height,1)):
                yPos = np.random.randint(0,high=self.boardSize-height)

            self.obstacleList += [Obstacle((height,width), (xPos,yPos), self.boardSize)]

    def collideWithRobotOrGoal(self, xPos, witdth, xy):

        return self.collide(self.robotPos[xy], self.robotShape[xy], xPos, witdth) or self.collide(self.goalPos[xy], self.goalShape[xy], xPos, witdth)

    def collide(self, pos1, shape1, pos2, shape2):

        return (pos2 < pos1 and pos1 < pos2 + shape2) or (pos2 < pos1 + shape1 and pos1 + shape1 < pos2 + shape2)

    def update(self, robotMove):

        self.robotPos = (int(self.robotPos[0] + robotMove[0]), int(self.robotPos[1] + robotMove[1]))

    def drawObstacles(self,board):

        for obstacle in self.obstacleList:

            obstacle.drawOnBoard(board)

    def drawBoard(self):

        board1 = np.zeros((self.boardSize,self.boardSize))
        board2 = np.zeros((self.boardSize, self.boardSize))
        board3 = np.zeros((self.boardSize, self.boardSize))
        self.drawObstacles(board1)
        board2[self.robotPos[0]:self.robotPos[0]+self.robotShape[0], self.robotPos[1]:self.robotPos[1]+self.robotShape[1]] = 1
        board3[self.goalPos[0]:self.goalPos[0]+self.goalShape[0],self.goalPos[1]:self.goalPos[1]+self.goalShape[1]] = 1

        return np.dstack((board1,board2,board3))





