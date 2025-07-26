import os
import shutil
from multiprocessing import Pool
from tqdm import tqdm


def move_subfolder(subfolder_path, target_folder):
    # 检查路径是否为文件夹
    if os.path.isdir(subfolder_path):
        # 移动子文件夹到目标文件夹B下
        shutil.copytree(subfolder_path, target_folder, dirs_exist_ok=True)


def not_split(folder):
    # 获取子文件夹列表
    subfolders = os.listdir(folder)

    # 遍历子文件夹
    for subfolder in tqdm(subfolders, total=len(subfolders), leave=True):
        # 构建子文件夹的完整路径
        subfolder_path = os.path.join(folder, subfolder)

        # 构建目标文件夹B下的子文件夹路径
        target_folder = os.path.join(folder_not_split_dir, subfolder)

        # 移动子文件夹
        move_subfolder(subfolder_path, target_folder)
    return True


# 定义数据集文件夹A和目标文件夹B的路径
folder_split_dir = '../data/AVEC2013_openface_align_224_split'
folder_not_split_dir = '../data/AVEC2013_openface_align_224_not_split'
num_process = 1  # 设置使用的进程数量

# 获取A文件夹下的train、dev、test文件夹路径
train_folder = os.path.join(folder_split_dir, 'train')
dev_folder = os.path.join(folder_split_dir, 'dev')
test_folder = os.path.join(folder_split_dir, 'test')

# 创建目标文件夹B（如果不存在）
os.makedirs(folder_not_split_dir, exist_ok=True)

if __name__ == '__main__':

    # 创建进程池
    with Pool(processes=num_process) as pool:
        # 使用进程池异步移动子文件夹
        results = []
        for folder in [train_folder, dev_folder, test_folder]:
            result = pool.apply_async(not_split, args=(folder,))
            results.append(result)

        # 等待所有进程完成
        for result in results:
            result.get()
