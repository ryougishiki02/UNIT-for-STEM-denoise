#    python all.py

import subprocess
import shutil
import os

source1 = 'divide/output'
destination1 = 'inputs/batch_folder'
source2 = 'results/batch_results'
destination2 = 'combine/inputs'

# 清空目标目录中的所有文件和子目录，但不删除目录本身
for filename in os.listdir(source1):
    file_path = os.path.join(source1, filename)
    if os.path.isdir(file_path):
        shutil.rmtree(file_path)  # 删除子文件夹
    else:
        os.remove(file_path)  # 删除文件

subprocess.run(['python', 'divide.py'])
# 清空目标目录中的所有文件和子目录，但不删除目录本身
for filename in os.listdir(destination1):
    file_path = os.path.join(destination1, filename)
    if os.path.isdir(file_path):
        shutil.rmtree(file_path)  # 删除子文件夹
    else:
        os.remove(file_path)  # 删除文件

for filename in os.listdir(source1):
    source_file = os.path.join(source1, filename)
    if os.path.isfile(source_file):
        destination_file = os.path.join(destination1, filename)
        shutil.move(source_file, destination_file)

# 定义命令及其参数
command = [
    'python', 'test_batch.py',
    '--trainer', 'UNIT',
    '--config', 'configs/unit_STEM_folder.yaml',
    '--input_folder', 'inputs/batch_folder',
    '--output_folder', 'results/batch_results',
    '--checkpoint', 'models/unit_STM.pt',
    '--a2b', '0'
]

# 执行命令
subprocess.run(command, check=True)

for filename in os.listdir(destination2):
    file_path = os.path.join(destination2, filename)
    if os.path.isdir(file_path):
        shutil.rmtree(file_path)  # 删除子文件夹
    else:
        os.remove(file_path)  # 删除文件

for filename in os.listdir(source2):
    source_file = os.path.join(source2, filename)

    # 确保是文件且文件名首个字符是数字
    if os.path.isfile(source_file) and filename[0].isdigit():
        destination_file = os.path.join(destination2, filename)
        shutil.move(source_file, destination_file)

subprocess.run(['python', 'combine.py'])



