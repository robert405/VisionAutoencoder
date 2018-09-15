import numpy as np
import cv2
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

def learningSchedule(t1, t):

    return 1 / (t1 + (t*10))