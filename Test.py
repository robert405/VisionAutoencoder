import cv2
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


engine = Engine(1,(15,15),(15,15),224)
boards = engine.drawAllBoard()

board = boards[0]**2
board[board > 0] = 255
boardEx = np.expand_dims(board, axis=2)
boardEx = np.repeat(boardEx, 3, axis=2).astype('uint8')

edged = cv2.Canny(boardEx, 127, 127)

imgList = []
imgList += [board]
imgList += [edged]

showImgs(imgList, 1, len(imgList))

