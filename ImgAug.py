import numpy as np

def addNoise(boards, scale):

    noise = np.random.normal(loc=0.0, scale=scale, size=boards.shape)

    return boards + noise

def dropOut(boards, prob):

    mask = np.random.binomial(1, (1-prob), size=boards.shape)

    return boards * mask

def coarseDropOut(boards, nb, size):

    shape = boards.shape
    rand1 = np.random.randint(0, high=shape[1]-size, size=nb)
    rand2 = np.random.randint(0, high=shape[1]-size, size=nb)

    canvas = np.ones((shape[1], shape[2]))
    for i in range(nb):
        canvas[rand1[i]:rand1[i]+size,rand2[i]:rand2[i]+size] = 0

    mask = np.expand_dims(canvas, axis=0)
    mask = np.repeat(mask, shape[0], axis=0)

    return boards * mask

def imgAug(boards):

    choice = np.random.randint(0,high=4)

    if (choice == 0):
        boards = addNoise(boards, 0.001)
    elif (choice == 1):
        boards = dropOut(boards, 0.2)
    elif (choice == 2):
        boards = coarseDropOut(boards, 10, 14)

    return boards

