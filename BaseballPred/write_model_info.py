import torch.nn as nn

def to_format_file(model: nn.Module, file_name: str):
    with open(file_name, "w") as f:
        for name, module in model.named_modules():
            if name == "":
                continue
            label_type = module.__class__.__name__
            f.write(f"{name}-{label_type}\n")