import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import json
import shutil
import os

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_SRC_PATH = BASE_DIR / 'dataset' / 'color'
DATA_PATH = BASE_DIR / 'app' / 'data' / 'classes.json'
IMAGE_DEST_PATH = BASE_DIR / 'app' / 'data' / 'images'

# remove all :Zone.Identifier files
#
#for item in IMAGE_SRC_PATH.iterdir():
#    for file in item.iterdir():
#        if file.name.__contains__(":Zone.Identifier"):
#            os.remove(file)


# loop through dataset and write json to a file with this structure 
# 
# plant type 1 :
#   health condition 1 : number of pictures
#   health condition 2 : ...
#   ...
# plant type 2 :
#   ...
#
#data = {}
#for item in IMAGE_SRC_PATH.iterdir():
#
#    name, status  = item.name.split('___')          # get name and status from folder name
#    if not data.get(name):
#        data[name] = {}                             # create dict for plant type if not exists 
#    data[name][status] = len(list(item.iterdir()))  # count number of images in folder and save it
#
#DATA_PATH.write_text(json.dumps(data, indent=2))


# for every case save 3 images in folders created in IMAGE_DEST_PATH / correct folder
#
#for item in IMAGE_SRC_PATH.iterdir():
#    dest_dir = IMAGE_DEST_PATH / item.name
#    dest_dir.mkdir(parents=True, exist_ok=True)
#    copied = 0
#    for file in item.iterdir():
#        if copied >= 3:
#           break
#        if not file.is_file():
#            continue
#        if ":Zone.Identifier" in file.name:
#            continue
#        try:
#            shutil.copy(file, dest_dir / file.name)
#            copied += 1
#        except Exception:
#            continue


