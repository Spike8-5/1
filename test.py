import os

root = "data/DIV2K_VAL/HR/"
for filename in os.listdir(root):
    img_path = os.path.join(root, filename)
    print(img_path)
    break
