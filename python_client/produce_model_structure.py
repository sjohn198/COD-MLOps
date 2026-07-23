from torch import nn
import torch
from win_predictor import WinPredictor
import json

if __name__ == "__main__":
    model = WinPredictor()

    sd = model.state_dict()
    layer_shapes = []
    for i, tensor in enumerate(list(sd.values())):
        if i % 2 == 0:
            #print(list(tensor.shape)[::-1])
            layer_shapes.append(list(tensor.shape)[::-1])

    with open("network_map.json", "w") as f:
        json.dump(layer_shapes, f)
    
    print("model structure produced.")