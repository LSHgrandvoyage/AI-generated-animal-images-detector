import os
import random
import shutil
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from tqdm import tqdm
import torch
from dotenv import load_dotenv

load_dotenv()

device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
base_dir = os.getenv("DATA_PATH")
classes = ["elephant", "cat", "chicken", "cow", "dog", "horse", "rabbit", "sheep"]
num_images_per_class = 100
size = (224, 224)
split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

print("Loading Stable Diffusion models...")

sd15 = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    dtype=torch.float16,
    variant="fp16"
).to(device)

sdxl = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    dtype=torch.float16
).to(device)

print("Models loaded successfully.")

generators = {
    "stable_v15": sd15,
    "sdxl": sdxl
}

temp_dir = os.path.join(base_dir, "temp_ai_images")

for generator_name, pipe in generators.items():
    print(f"Generating images with {generator_name} ...")
    for cls in classes:
        temp_save_dir = os.path.join(temp_dir, generator_name, cls)
        os.makedirs(temp_save_dir, exist_ok=True)
        for i in tqdm(range(num_images_per_class), desc=f"{generator_name}-{cls}"):
            prompt = f"photo of a {cls}, high quality, natural lighting"
            image = pipe(prompt, num_inference_steps=30).images[0]
            image = image.resize(size)
            image.save(os.path.join(temp_save_dir, f"{cls}_{i:04d}.png"))

print("\nSplitting generated images into train, val, and test sets...")

for generator_name in generators.keys():
    for cls in classes:
        temp_class_dir = os.path.join(temp_dir, generator_name, cls)
        images = [f for f in os.listdir(temp_class_dir) if f.endswith('.png')]
        random.shuffle(images)
        train_end = int(len(images) * split_ratios["train"])
        val_end = train_end + int(len(images) * split_ratios["val"])
        train_files = images[:train_end]
        val_files = images[train_end:val_end]
        test_files = images[val_end:]
        for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            dest_dir = os.path.join(base_dir, split_name, "ai", generator_name, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for file_name in files:
                src_path = os.path.join(temp_class_dir, file_name)
                dest_path = os.path.join(dest_dir, file_name)
                shutil.move(src_path, dest_path)

print("\nCleaning up temporary files...")
shutil.rmtree(temp_dir)
print("\nAll tasks are complete! Dataset is ready.")
