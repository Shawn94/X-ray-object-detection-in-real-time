from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

path = "raw_images/P2_10001.raw"
with open(path, 'r') as infile:
    infile.seek(0)
    data = np.fromfile(infile, dtype='<H').reshape(680, 680)

plt.imshow(data, cmap='gray')
plt.axis('off')
plt.show()

