import torch
from PytorchNetwork import VisionEncoder, VisionDecoder, PositionEstimator
from TrainingFunction import train
from EvaluationFunction import evaluateModel
import matplotlib.pyplot as plt

saveIt = -1
encoderSavePath = "../SavedModel/VisionEncoderModel"
decoderSavePath = "../SavedModel/VisionDecoderModel"
posEstSavePath = "../SavedModel/PositionEstimatorModel"

visionEncoderModel = VisionEncoder().cuda()
visionDecoderModel = VisionDecoder().cuda()
positionEstimator = PositionEstimator().cuda()

if (saveIt >= 0):
    visionEncoderModel.load_state_dict(torch.load(encoderSavePath + str(saveIt)))
    visionDecoderModel.load_state_dict(torch.load(decoderSavePath + str(saveIt)))
    positionEstimator.load_state_dict(torch.load(posEstSavePath + str(saveIt)))

nbIteration = 15000
batchSize = 50
#lr = 1e-4
t1 = 10000


lossList, lossList2 = train(visionEncoderModel, visionDecoderModel, positionEstimator, nbIteration, batchSize, t1)

torch.save(visionEncoderModel.state_dict(), encoderSavePath + str(saveIt + 1))
torch.save(visionDecoderModel.state_dict(), decoderSavePath + str(saveIt + 1))
torch.save(positionEstimator.state_dict(), posEstSavePath + str(saveIt + 1))

plt.plot(lossList)
plt.show()

plt.plot(lossList2)
plt.show()

evaluateModel(visionEncoderModel, visionDecoderModel)