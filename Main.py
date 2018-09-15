import torch
from PytorchNetwork import VisionEncoder, VisionDecoder, PositionEstimator, VisionEdgeDecoder
from TrainingFunction import train
from EvaluationFunction import evaluateModel
import matplotlib.pyplot as plt

saveIt = -1
encoderSavePath = "../SavedModel/VisionEncoderModel"
decoderSavePath = "../SavedModel/VisionDecoderModel"
posEstSavePath = "../SavedModel/PositionEstimatorModel"
edgeDecoderSavePath = "../SavedModel/VisionEdgeDecoderModel"

visionEncoderModel = VisionEncoder().cuda()
visionDecoderModel = VisionDecoder().cuda()
positionEstimator = PositionEstimator().cuda()
visionEdgeDecoderModel = VisionEdgeDecoder().cuda()

if (saveIt >= 0):
    visionEncoderModel.load_state_dict(torch.load(encoderSavePath + str(saveIt)))
    visionDecoderModel.load_state_dict(torch.load(decoderSavePath + str(saveIt)))
    positionEstimator.load_state_dict(torch.load(posEstSavePath + str(saveIt)))
    visionEdgeDecoderModel.load_state_dict(torch.load(edgeDecoderSavePath + str(saveIt)))

nbIteration = 100
batchSize = 50
#lr = 1e-4
t1 = 10000

multitask = {'autoEncoder':True, 'posEstimator':True, 'edgeDecoder':True}

lossList, lossList2, lossList3 = train(visionEncoderModel, visionDecoderModel, positionEstimator, visionEdgeDecoderModel, nbIteration, batchSize, t1, multitask)

torch.save(visionEncoderModel.state_dict(), encoderSavePath + str(saveIt + 1))
torch.save(visionDecoderModel.state_dict(), decoderSavePath + str(saveIt + 1))
torch.save(positionEstimator.state_dict(), posEstSavePath + str(saveIt + 1))
torch.save(visionEdgeDecoderModel.state_dict(), edgeDecoderSavePath + str(saveIt + 1))

plt.plot(lossList)
plt.show()

plt.plot(lossList2)
plt.show()

plt.plot(lossList3)
plt.show()

evaluateModel(visionEncoderModel, visionDecoderModel, visionEdgeDecoderModel)