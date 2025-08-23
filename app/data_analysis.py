import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / 'dataset' / 'color'
DATA_PATH = BASE_DIR / 'app' / 'data' / 'classes.json'

#
# loop through dataset and write json to a file with this structure
#
# plant type 1 :
#   health condition 1 : number of pictures
#   health condition 2 : ...
#   ...
# plant type 2 :
#   ...
#
data = {}
for item in IMAGE_PATH.iterdir():
    name, status  = item.name.split('___')
    if not data.get(name):
        data[name] = {}
    data[name][status] = len(list(item.iterdir()))

DATA_PATH.write_text(json.dumps(data, indent=2))