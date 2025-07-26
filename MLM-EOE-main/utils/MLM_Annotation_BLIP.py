import os
import random
import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from transformers import BlipProcessor, BlipForConditionalGeneration

num_workers = 16
batch_size = 8
gpu_id = 0
dataset = "AVEC2014"
model_name = "blip"
image_interval = 1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_device(gpu_id):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: cuda:{gpu_id}")
    else:
        device = torch.device('cpu')
        print("CUDA is not available, using CPU.")
    return device

def load_model_and_processor(device):
    processor = BlipProcessor.from_pretrained(
        "")
    model = BlipForConditionalGeneration.from_pretrained(
        "").to(device)
    model.eval()
    return model, processor

def get_autocast_context(device):
    return torch.cuda.amp.autocast if device.type == 'cuda' else lambda: torch.no_grad()

def match_image_folders_with_csv(image_dir, image_interval):
    image_folders = sorted([os.path.join(image_dir, folder) for folder in os.listdir(image_dir) if
                            os.path.isdir(os.path.join(image_dir, folder))])

    image_paths_dict = []
    for image_folder in image_folders:
        image_files = sorted(
            [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg')])
        image_files = image_files[::image_interval]
        image_paths_dict.append(image_files)
    return image_paths_dict

def calculate_similarity_batch(image_batch, text_tokenize, model, device, processor, texts_score, scaler):
    images = [Image.open(img_path).convert('RGB') for img_path in image_batch]
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    pixel_values = inputs["pixel_values"]

    with torch.no_grad(), scaler():
        generated_ids = model.generate(pixel_values=pixel_values)
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    similarity_scores_batch = []
    for text in text_tokenize:
        text_inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, input_ids=text_inputs.input_ids)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            max_probs = np.max(probs, axis=-1)
        similarity_scores_batch.append(max_probs)

    similarity_scores_batch = np.array(similarity_scores_batch).T
    normalized_scores_batch = similarity_scores_batch / similarity_scores_batch.sum(axis=1, keepdims=True)
    result_scores_batch = np.dot(normalized_scores_batch, texts_score)
    return result_scores_batch

def prepare_text_features():
    texts = ["ecstatic", "happiness", "satisfaction", "calm", "anxiety", "sadness", "anger"]
    return texts

def process_folder_parallel(image_folder, text_tokenize, model, device, processor, texts_score, scaler):
    results = []
    image_batches = [image_folder[i:i + batch_size] for i in range(0, len(image_folder), batch_size)]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        result_probs = list(tqdm(executor.map(
            calculate_similarity_batch,
            image_batches,
            [text_tokenize] * len(image_batches),
            [model] * len(image_batches),
            [device] * len(image_batches),
            [processor] * len(image_batches),
            [texts_score] * len(image_batches),
            [scaler] * len(image_batches)),
            total=len(image_batches), desc="Processing Images in Folder", leave=False))

        results.extend(result_probs)
    return np.concatenate(results)

def main(image_directory, gpu_id, image_interval, seed=42):
    set_seed(seed)
    device = set_device(gpu_id)
    model, processor = load_model_and_processor(device)
    scaler = get_autocast_context(device)
    image_paths = match_image_folders_with_csv(image_directory, image_interval)
    text_tokenize = prepare_text_features()
    texts_score = np.array([0, 1, 2, 3, 4, 5, 6])

    for image_folder in tqdm(image_paths, desc="Processing Image Folders"):
        folder_name = os.path.basename(os.path.dirname(image_folder[0])).rsplit('_', 1)[0]
        output_dir = f""
        output_csv = os.path.join(output_dir, f"{folder_name}.csv")
        if os.path.exists(output_csv):
            print(f"Skipping {folder_name}, already processed.")
            continue

        folder_result = process_folder_parallel(image_folder, text_tokenize, model, device, processor, texts_score, scaler)
        df = pd.DataFrame(folder_result, index=[os.path.basename(path) for path in image_folder])
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_csv, header=False)
        print(f"Processed and saved results for {folder_name}.")

if __name__ == "__main__":
    image_directory = f''
    main(image_directory, gpu_id, image_interval, seed=42)