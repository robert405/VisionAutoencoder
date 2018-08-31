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

    with torch.no_grad():
        while (True):

            engine = Engine(1, (15, 15), (15, 15), 224)
            boards = engine.drawAllBoard()
            originalBoard = boards[0] + 10
            originalBoard = originalBoard * (255 / 20)

            torchBoards = torch.FloatTensor(boards).cuda()
            torchBoards = torch.unsqueeze(torchBoards, 1)
            torchBoards = torchBoards / 10

            features = visionEncoderModel(torchBoards)
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