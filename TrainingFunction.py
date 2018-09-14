import torch
import torch.nn as nn
import numpy as np
import cv2
import time
from Engine import Engine
from ImgAug import imgAug

def getEdge(boards):

    edges = np.zeros_like(boards)

    for i in range(boards.shape[0]):

        board = boards[i] ** 2
        board[board > 0] = 255
        boardEx = np.expand_dims(board, axis=2)
        boardEx = np.repeat(boardEx, 3, axis=2).astype('uint8')
        edges[i] = cv2.Canny(boardEx, 127, 127)

    edges = edges - 127.5
    edges = edges / 128

    return edges

def lerningSchedule(t1, t):

    return 1 / (t1 + (t*10))

def train(visionEncoderModel, visionDecoderModel, positionEstimator, visionEdgeDecoderModel, nbIteration, batchSize, t1, multitask):

    print("Starting trainning!")
    lr = lerningSchedule(t1, 0)

    criterion = nn.MSELoss()

    optimizer = None
    optimizer2 = None
    optimizer3 = None
    lossList = []
    lossList2 = []
    lossList3 = []
    moduloPrint = 100
    visionEncoderModel.train()
    visionDecoderModel.train()
    positionEstimator.train()
    visionEdgeDecoderModel.train()
    meanLoss = 0
    meanLoss2 = 0
    meanLoss3 = 0

    if (multitask['autoEncoder']):
        optimizer = torch.optim.Adam(list(visionEncoderModel.parameters()) + list(visionDecoderModel.parameters()), lr=lr)

    if (multitask['posEstimator']):
        optimizer2 = torch.optim.Adam(list(visionEncoderModel.parameters()) + list(positionEstimator.parameters()), lr=lr)

    if (multitask['edgeDecoder']):
        optimizer3 = torch.optim.Adam(list(visionEncoderModel.parameters()) + list(visionEdgeDecoderModel.parameters()),lr=lr)

    start = time.time()

    resolution = 224
    pixelRange = np.arange(resolution)

    row = np.expand_dims(pixelRange, axis=1)
    row = np.repeat(row, resolution, axis=1)
    row = np.expand_dims(row, axis=0)
    row = np.repeat(row, batchSize, axis=0)
    row = np.expand_dims(row, axis=3)
    row = row / resolution

    column = np.expand_dims(pixelRange, axis=0)
    column = np.repeat(column, resolution, axis=0)
    column = np.expand_dims(column, axis=0)
    column = np.repeat(column, batchSize, axis=0)
    column = np.expand_dims(column, axis=3)
    column = column / resolution

    for k in range(nbIteration):

        engine = Engine(batchSize,(15,15),(15,15),224)
        boards = engine.drawAllBoard()
        inputBoards = boards / 10
        inputBoards = imgAug(inputBoards)
        inputBoards = np.expand_dims(inputBoards, axis=3)
        inputBoards = np.concatenate((inputBoards, row, column), axis=3)
        torchInputBoards = torch.FloatTensor(inputBoards).cuda()
        torchInputBoards = torchInputBoards.permute(0, 3, 1, 2)

        if (multitask['autoEncoder']):

            torchBoards = torch.FloatTensor(boards).cuda()
            torchBoards = torch.unsqueeze(torchBoards, 1)

            features = visionEncoderModel(torchInputBoards)
            pred = visionDecoderModel(features)

            loss = calculateLoss(pred, torchBoards)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            meanLoss += loss.data.cpu().numpy()

        if (multitask['posEstimator']):

            robotPos = engine.getAllRobotPos()
            goalPos = engine.getAllGoalPos()
            dist = engine.getDist(robotPos)
            dist = np.expand_dims(dist, axis=1)
            allPosition = np.concatenate((robotPos, goalPos), axis=1)
            allPositionAndDist = np.concatenate((allPosition, dist), axis=1)
            torchPositionAndDist = torch.FloatTensor(allPositionAndDist).cuda()
            halfResolution = resolution / 2
            torchPositionAndDist = torchPositionAndDist - halfResolution
            torchPositionAndDist = torchPositionAndDist / halfResolution

            features2 = visionEncoderModel(torchInputBoards)

            posPred = positionEstimator(features2)
            loss2 = criterion(posPred, torchPositionAndDist)
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            meanLoss2 += loss2.data.cpu().numpy()

        if (multitask['edgeDecoder']):

            edges = getEdge(boards)
            torchEdgeBoards = torch.FloatTensor(edges).cuda()
            torchEdgeBoards = torch.unsqueeze(torchEdgeBoards, 1)

            features3 = visionEncoderModel(torchInputBoards)

            edgePred = visionEdgeDecoderModel(features3)
            loss3 = criterion(edgePred, torchEdgeBoards)
            optimizer3.zero_grad()
            loss3.backward()
            optimizer3.step()

            meanLoss3 += loss3.data.cpu().numpy()

        if ((k+1) % moduloPrint == 0):

            lr = lerningSchedule(t1, k)
            msg = "Iteration : " + str(k + 1) + " / " + str(nbIteration)

            if (multitask['autoEncoder']):
                meanLoss = meanLoss / moduloPrint
                lossList += [meanLoss]
                msg += ", Current mean loss : " + str(meanLoss)
                meanLoss = 0
                optimizer = torch.optim.Adam(list(visionEncoderModel.parameters()) + list(visionDecoderModel.parameters()), lr=lr)

            if (multitask['posEstimator']):
                meanLoss2 = meanLoss2 / moduloPrint
                lossList2 += [meanLoss2]
                msg += ", Current mean loss 2 : " + str(meanLoss2)
                meanLoss2 = 0
                optimizer2 = torch.optim.Adam(list(visionEncoderModel.parameters()) + list(positionEstimator.parameters()), lr=lr)

            if (multitask['edgeDecoder']):
                meanLoss3 = meanLoss3 / moduloPrint
                lossList3 += [meanLoss3]
                msg += ", Current mean loss 3 : " + str(meanLoss3)
                meanLoss3 = 0
                optimizer3 = torch.optim.Adam(list(visionEncoderModel.parameters()) + list(visionEdgeDecoderModel.parameters()), lr=lr)

            print(msg)

        if (k % 200 == 0):
            end = time.time()
            timeTillNow = end - start
            predictedRemainingTime = (timeTillNow / (k + 1)) * (nbIteration - (k + 1))
            print("--------------------------------------------------------------------")
            print("Time to run since started (sec) : " + str(timeTillNow))
            print("Predicted remaining time (sec) : " + str(predictedRemainingTime))
            print("--------------------------------------------------------------------")

    end = time.time()
    print("Time to run in second : " + str(end - start))

    return lossList, lossList2, lossList3
