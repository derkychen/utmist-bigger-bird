import torch
import torch.nn as nn

from transformers import BigBirdModel, BigBirdConfig

# For now just run Big Bird as it is
config = BigBirdConfig()

model = BigBirdModel(config)

print(model)  # test

