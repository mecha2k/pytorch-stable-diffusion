import torch
import torch.nn as nn

from sd import encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else device)
print(f"{device} is available in torch and cuda version : {torch.version.cuda}")


if __name__ == "__main__":
    inputs = torch.randn(64, 3, 32, 32).to(device)
    noise = torch.randn(1).to(device)
    model = encoder.VAE_Encoder().to(device)
    outputs = model(inputs, noise)
    print(outputs.shape)
