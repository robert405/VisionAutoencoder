import torch
import torch.nn.functional as F
import numpy as np
from Engine import Engine
import matplotlib.pyplot as plt

def showImgs(imgs, nbEx, nbCl):
    counter = 0
    for i in range(nbCl):
        for j in range(nbEx):
            plt.subplot(nbEx, nbCl, counter+1)
            plt.imshow(imgs[counter].astype('uint8'))
            plt.axis('off')
            counter += 1

    plt.show()

def evaluateModel(visionEncoderModel, visionDecoderModel):

    visionEncoderModel.eval()
    visionDecoderModel.eval()

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
            originalBoard = boards[0] + 10
            originalBoard = originalBoard * (255 / 20)

            inputBoards = np.expand_dims(boards, axis=3)
            inputBoards = inputBoards / 10
            inputBoards = np.concatenate((inputBoards, row, column), axis=3)
            torchInputBoards = torch.FloatTensor(inputBoards).cuda()
            torchInputBoards = torchInputBoards.permute(0, 3, 1, 2)

            features = visionEncoderModel(torchInputBoards)
            pred = visionDecoderModel(features)
            pred = pred.squeeze()
            pred = F.sigmoid(pred)
            pred = pred + 10.0
            pred = pred * (255 / 20)

            compareList = [originalBoard, pred.data.cpu().numpy()]

            showImgs(compareList, 1, 2)

            print("exit or continu?")
            answer = input()

            if (answer == "exit"):
                break