import os

path = "dataset/train"

for folder in os.listdir(path):
    print(folder, "→", len(os.listdir(os.path.join(path, folder))))