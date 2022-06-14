import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def convert(path):

    PATH = os.path.join(os.getcwd(), path)
    files = os.listdir(PATH)

    for i in (files):

        input_filename = os.path.join(PATH,i)
        shape = (1536, 1536) # matrix size
        output_filename = input_filename.split('\\')[-1].split('.')[0] + '.png'

        # Read the image file
        img= open(input_filename, 'rb')
        data = np.fromfile(input_filename,  dtype=np.uint16)
        shape = int(np.sqrt(data.shape[0]))
        image = data.reshape(shape,shape)

        #Rescale the values to 0-255
        _min = np.amin(image)
        _max = np.amax(image)
        image = (image - _min) * 255.0 / (_max - _min)
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
    