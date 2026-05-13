import torch
import torch.nn as nn

from transformers import (
    BigBirdModel,
    BigBirdConfig
)

# For now, just running BigBird as it is
config = BigBirdConfig()

model = BigBirdModel(config)

print(model) # test
