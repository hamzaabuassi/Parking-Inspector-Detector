# calibrate_slots_matplotlib.py

import matplotlib.pyplot as plt
import pickle

IMAGE_PATH = "carParkImg.png"  
NUM_SLOTS  = 69                  
OUTPUT_PKL = "CarParkPos.pkl"

img = plt.imread(IMAGE_PATH)
fig, ax = plt.subplots(figsize=(10,8))
ax.imshow(img)
ax.set_title(f"Click {NUM_SLOTS} top-left corners, then close window")
points = plt.ginput(NUM_SLOTS, timeout=0)  
plt.close(fig)

coords = [(int(x), int(y)) for x, y in points]
with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump(coords, f)
print(f"Saved {len(coords)} slots to {OUTPUT_PKL}")
