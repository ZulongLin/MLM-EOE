import os
import csv
import shutil

csv_file = '../data/AVEC2013record.csv'
output_folder = '../data/AVEC2013_openface_align_224_split/AVEC2013_labels'

# 创建存储标签的文件夹
os.makedirs(output_folder, exist_ok=True)

with open(csv_file, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        filename = row['filename']
        labelscore = row['labelscore']

        # 提取文件名前五个字符作为新文件的名称
        new_filename = filename[:5] + '_Depression' + '.csv'

        # 构建新文件的路径
        new_file_path = os.path.join(output_folder, new_filename)

        # 将样本标签写入新文件
        with open(new_file_path, 'w', newline='') as new_file:
            writer = csv.writer(new_file)
            writer.writerow([labelscore])

print("处理完成")
