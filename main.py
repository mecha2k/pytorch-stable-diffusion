import torch
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer

from sd import pipeline, model_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else device)
print(f"{device} is available in torch and cuda version : {torch.version.cuda}")


model_file = "data/v1-5-pruned-emaonly.ckpt"
tokenizer = CLIPTokenizer(
    "data/tokenizer_vocab.json",
    merges_file="data/tokenizer_merges.txt",
    clean_up_tokenization_spaces=True,
)
models = model_loader.preload_models_from_standard_weights(model_file, device)


## TEXT TO IMAGE
prompt = "A dog with blue sunglasses, full body, yellow hat, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
# prompt = "A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
uncond_prompt = ""  # Also known as negative prompt
do_cfg = True
cfg_scale = 8  # min: 1, max: 14

## IMAGE TO IMAGE
# input_image = None
image_path = "images/dog.png"
input_image = Image.open(image_path)

# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
strength = 0.9

## SAMPLER
sampler = "ddpm"
num_inference_steps = 50
seed = 42

output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=device,
    idle_device="cpu",
    tokenizer=tokenizer,
)

# Combine the input image and the output image into a single image.
img = Image.fromarray(output_image)
img.save("images/output.png")
