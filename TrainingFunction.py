import torch
import torch.nn as nn
import numpy as np
import time
from Engine import Engine

def train(visionEncoderModel, visionDecoderModel, positionEstimator, nbIteration, batchSize, lr):

    print("Starting trainning!")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(visionEncoderModel.parameters()) + list(visionDecoderModel.parameters()), lr=lr)
    optimizer2 = torch.optim.Adam(list(visionEncoderModel.parameters()) + list(positionEstimator.parameters()), lr=lr)

    lossList = []
    lossList2 = []
    moduloPrint = 25
    visionEncoderModel.train()
    visionDecoderModel.train()
    positionEstimator.train()
    meanLoss = 0
    meanLoss2 = 0
    start = time.time()

    for k in range(nbIteration):

        engine = Engine(batchSize,(15,15),(15,15),224)
        boards = engine.drawAllBoard()

        robotPos = engine.getAllRobotPos()
        goalPos = engine.getAllGoalPos()
        dist = engine.getDist(robotPos)
        dist = np.expand_dims(dist, axis=1)
        allPosition = np.concatenate((robotPos, goalPos), axis=1)
        allPositionAndDist = np.concatenate((allPosition, dist), axis=1)
        torchPositionAndDist = torch.FloatTensor(allPositionAndDist).cuda()

        torchBoards = torch.FloatTensor(boards).cuda()
        torchBoards = torch.unsqueeze(torchBoards, 1)
        torchInputBoards = torchBoards / 10

        features = visionEncoderModel(torchInputBoards)
        pred = visionDecoderModel(features)

        loss = criterion(pred, torchBoards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meanLoss += loss.data.cpu().numpy()

        features2 = visionEncoderModel(torchInputBoards)

        posPred = positionEstimator(features2)
        loss2 = criterion(posPred, torchPositionAndDist)
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        meanLoss2 += loss2.data.cpu().numpy()

        if ((k+1) % moduloPrint == 0):
            meanLoss = meanLoss / moduloPrint
            lossList += [meanLoss]
            meanLoss2 = meanLoss2 / moduloPrint
            lossList2 += [meanLoss2]
            print("Iteration : " + str(k+1) + " / " + str(nbIteration) + ", Current mean loss : " + str(meanLoss) + ", Current mean loss 2 : " + str(meanLoss2))
            meanLoss = 0
            meanLoss2 = 0

        if (k % 200 == 0):
            end = time.time()
            timeTillNow = end - start
            predictedRemainingTime = (timeTillNow / (k + 1)) * (nbIteration - (k + 1))
            print("--------------------------------------------------------------------")
            print("Time to run since started (sec) : " + str(timeTillNow))
            print("Predicted remaining time (sec) : " + str(predictedRemainingTime))
            print("--------------------------------------------------------------------")

        if (k % 1000 == 0):
            lr = lr * 0.4
            optimizer = torch.optim.Adam(list(visionEncoderModel.parameters()) + list(visionDecoderModel.parameters()),lr=lr)
            optimizer2 = torch.optim.Adam(list(visionEncoderModel.parameters()) + list(positionEstimator.parameters()),lr=lr)

    end = time.time()
    print("Time to run in second : " + str(end - start))

    return lossList, lossList2
