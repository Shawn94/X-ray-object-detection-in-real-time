from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

path = "raw_images/raw_image.raw"
with open(path, 'r') as rawdata:
    rawdata.seek(0)
    data = np.fromfile(rawdata, dtype='>H').reshape(680, 680)

plt.imshow(data, cmap='gray')
plt.axis('off')
plt.show()

