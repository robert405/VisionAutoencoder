import torch
import torch.nn.functional as F
import numpy as np
from Engine import Engine
from Utils import showImgs, getEdge

def evaluateModel(visionEncoderModel, visionDecoderModel, visionEdgeDecoderModel):

    visionEncoderModel.eval()
    visionDecoderModel.eval()
    visionEdgeDecoderModel.eval()

    resolution = 224
    pixelRange = np.arange(resolution)

    row = np.expand_dims(pixelRange, axis=1)
    row = np.repeat(row, resolution, axis=1)
    row = np.expand_dims(row, axis=0)
    row = np.expand_dims(row, axis=3)
    row = row / resolution

    column = np.expand_dims(pixelRange, axis=0)
    column = np.repeat(column, resolution, axis=0)
    column = np.expand_dims(column, axis=0)
    column = np.expand_dims(column, axis=3)
    column = column / resolution

    with torch.no_grad():
        while (True):

            engine = Engine(1, (15, 15), (15, 15), 224)
            boards = engine.drawAllBoard()
            originalBoard = np.squeeze(boards) * 255

            edges = getEdge(boards)
            edge = edges[0] * 255

            inputBoards = np.concatenate((boards, row, column), axis=3)
            torchInputBoards = torch.FloatTensor(inputBoards).cuda()
            torchInputBoards = torchInputBoards.permute(0, 3, 1, 2)

            features = visionEncoderModel(torchInputBoards)

            pred = visionDecoderModel(features)
            pred = F.softmax(pred, dim=1)
            pred = pred.permute(0, 2, 3, 1)
            pred = pred.squeeze()
            predBoard = pred.data.cpu().numpy()
            predBoard = predBoard * 255
            decoderResult = predBoard[:, :, 0:3]

            edgePred = visionEdgeDecoderModel(features)
            edgePred = F.sigmoid(edgePred)
            edgePred = edgePred.squeeze()
            edgePred = edgePred * 255
            edgeResult = edgePred.data.cpu().numpy()

            compareList = [originalBoard, decoderResult, edge, edgeResult]

            showImgs(compareList, 2, 2)

            print("exit or continu?")
            answer = input()

            if (answer == "exit"):
                break