import multiprocessing
import os
import shutil
import csv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def copy_files(source, destination):
    for filename in os.listdir(source):
        file_path = os.path.join(source, filename)
        if os.path.isfile(file_path):
            shutil.copy(file_path, destination)


def process_folder(row, folder, output_path):
    train_path = os.path.join(output_path, 'train', folder)
    dev_path = os.path.join(output_path, 'dev', folder)
    test_path = os.path.join(output_path, 'test', folder)
    source_folder = os.path.join(folder_path, folder)
    if row['group'] == 'train':
        os.makedirs(train_path, exist_ok=True)
        destination = train_path
    elif row['group'] == 'dev':
        os.makedirs(dev_path, exist_ok=True)
        destination = dev_path
    elif row['group'] == 'test':
        os.makedirs(test_path, exist_ok=True)
        destination = test_path

    copy_files(source_folder, destination)

    return folder


# 设置输入参数
folder_path = '../data/AVEC2013_openface_align_224'
csv_file = '../data/AVEC2013_openface_align_224_split/2013_mini.csv'
output_path = '../data/AVEC2013_openface_align_224_split'
num_processes = 1  # 设置进程数

if __name__ == '__main__':
    with multiprocessing.Pool(processes=num_processes) as pool:
        with open(csv_file, 'r') as csvfile:
            results = []
            reader = csv.DictReader(csvfile)
            for row in reader:
                if (row['dataname'] == 'AVEC2014'):
                    break
                result = pool.apply_async(process_folder, (row, row['filename'], output_path))
                results.append(result)
        for result in tqdm(results, total=len(results), leave=True):
            result.get()
    print("All folders processed.")
