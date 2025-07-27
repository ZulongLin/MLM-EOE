import os
import pandas as pd
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

dataset = "AVEC2014"
model_name = "clip_vit"

def match_image_folders_with_csv(image_dir):
    image_folders = sorted([os.path.join(image_dir, folder) for folder in os.listdir(image_dir) if
                            os.path.isdir(os.path.join(image_dir, folder))])
    image_paths_dict = []

    for image_folder in image_folders[26:]:
        image_files = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) if
                              img.endswith('.jpg')])
        image_paths_dict.append(image_files)

    return image_paths_dict

def calculate_similarity(image_dir, text_tokenize, model, device, preprocess, texts_score):
    image = preprocess(Image.open(image_dir)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text_tokenize)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    result_score = np.round(np.dot(probs, texts_score), 4)
    return result_score

def prepare_text_features(device):
    texts = ["ecstatic", "happiness", "satisfaction", "calm", "anxiety", "sadness", "anger"]
    text_tokenize = clip.tokenize([f"The displayed emotion is {c}" for c in texts]).to(device)
    return text_tokenize

def process_folder_parallel(image_folder, text_tokenize, model, device, preprocess, texts_score):
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        result_probs = list(tqdm(executor.map(
            calculate_similarity,
            image_folder,
            [text_tokenize] * len(image_folder),
            [model] * len(image_folder),
            [device] * len(image_folder),
            [preprocess] * len(image_folder),
            [texts_score] * len(image_folder)),
            total=len(image_folder), desc="Processing Images in Folder", leave=False))
        results.extend(result_probs)
    return results

texts_score = np.array([6, 5, 4, 3, 2, 1, 0])

image_directory = f''
image_paths = match_image_folders_with_csv(image_directory)

gpu_id = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, preprocess = clip.load("", device=device)
text_tokenize = prepare_text_features(device)

for image_folder in tqdm(image_paths, desc="Processing Image Folders"):
    folder_result = process_folder_parallel(image_folder, text_tokenize, model, device, preprocess, texts_score)
    result = []
    for result_score in folder_result:
        result.append(result_score[0])

    df = pd.DataFrame(folder_result, index=[os.path.basename(path) for path in image_folder])
    folder_name = os.path.basename(os.path.dirname(image_folder[0])).rsplit('_', 1)[0]
    output_dir = f""
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"{folder_name}.csv")
    df.to_csv(output_csv, header=False)