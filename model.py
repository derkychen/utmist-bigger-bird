import torch
import torch.nn as nn

# from transformers import (
#     BigBirdModel,
#     BigBirdConfig
# )

# For now, just running BigBird as it is

# config = BigBirdConfig()

# model = BigBirdModel(config)

# Check the class that BigBird's attention model lives in
# print(type(model.encoder.layer[0].attention.self))

from transformers.models.big_bird.modeling_big_bird import (
    BigBirdBlockSparseAttention
)

class BiggerBirdAttention(BigBirdBlockSparseAttention):
    def __init__(self, config):
        super().__init__(config)
    
    
