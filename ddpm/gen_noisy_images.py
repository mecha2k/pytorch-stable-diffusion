import torch
import numpy as np
import math

from PIL import Image
from model import DiffusionModel

model = DiffusionModel((728 * 728), 1000, 3)

# How many noise levels to generate
noise_levels = [0, 10, 50, 75, 100, 250, 500, 750, 1000]

img = Image.open("Umar.png", formats=["png"])
img_tensor = torch.tensor(np.array(img))
img_tensor = ((img_tensor / 255.0) * 2.0) - 1.0
# Create a batch by repeating the same image many times
batch = img_tensor.repeat(len(noise_levels), 1, 1, 1)

ts = torch.tensor(noise_levels, dtype=torch.int, device=batch.device)
noise_imgs = []
epsilons = torch.randn(batch.shape, device=batch.device)
# Generate a noisifed version of the image for each noise level
for i in range(len(ts)):
    a_hat = model.alpha_bar(ts[i])
    noise_imgs.append((math.sqrt(a_hat) * batch[i]) + (math.sqrt(1 - a_hat) * epsilons[i]))


noise_imgs = torch.stack(noise_imgs, dim=0)
noise_imgs = (noise_imgs.clamp(-1, 1) + 1) / 2
noise_imgs = (noise_imgs * 255).type(torch.uint8)

# Convert back to image and display
display_img = Image.fromarray(noise_imgs[5].squeeze(0).numpy(), "RGB")
display_img.save("Umar_noise.png")
