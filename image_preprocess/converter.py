import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def convert(path):

    PATH = os.path.join(os.getcwd(), path)
    files = os.listdir(PATH)

    for i in (files):
        if i.endswith(".raw"):
            input_filename = os.path.join(PATH,i)
            output_filename = input_filename.split('\\')[-1].split('.')[0] + '.png'

            # Read the image file
            with open(input_filename, 'r') as rawdata:
                rawdata.seek(0) #offset
                #little-endian '<H', big-endian '>H'
                rawdata = np.fromfile(rawdata,  dtype='<H')
                shape = int(np.sqrt(rawdata.shape[0]))
                image_numpy = rawdata.reshape(shape,shape)

            #Rescale the values to 0-255
            _min = np.amin(image_numpy)
            _max = np.amax(image_numpy)
            image = (image_numpy - _min) * 255.0 / (_max - _min)
            image = np.uint8(image)

            #Brighten the image
            value = 200
            brightened_image = np.where((255 - image) < value,255,image+value)

            #create directory to save png images if not exist
            if not os.path.exists('converted'):
                os.makedirs('converted')

            # Save the files in to the converted directory
            path_to_save = os.path.join(os.getcwd()+'/converted',output_filename)
            plt.imsave(path_to_save, brightened_image, cmap='gray')
            print(f'Converted {i} file to png format')
            


if __name__ == "__main__":
    convert(sys.argv[1])
    