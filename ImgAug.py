import numpy as np

def addNoise(boards, scale):

    noise = np.random.normal(loc=0.0, scale=scale, size=boards.shape)

    return boards + noise

def dropOut(boards, prob):

    mask = np.random.binomial(1, (1-prob), size=boards.shape)

    return boards * mask

def imgAug(boards):

    choice = np.random.randint(0,high=3)

    if (choice == 0):
        boards = addNoise(boards, 0.001)
    elif (choice == 1):
        boards = dropOut(boards, 0.3)

    return boards
