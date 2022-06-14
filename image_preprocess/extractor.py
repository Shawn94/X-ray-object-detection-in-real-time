import os
import sys
import cv2

def extractImages(input, output,frame):
    
    #create output folder if not exist
    if not os.path.exists(output):
            os.mkdir(output)
    count = 1
    #read the video
    vidcap = cv2.VideoCapture(input)
    ret,cap = vidcap.read()
    if ret: 
        print("Found video file. Starting to extract images")
        while ret: 
            ret,cap = vidcap.read()
            if count % frame == 0: #save image by required frequncy rate
                print("Extracting image{:04d}.jpg".format(count//frame))
                cv2.imwrite( (output + "\\image{:04d}.jpg".format(count//frame)), cap) 
            count = count + 1   
    else: print("FileNotFound. Please check the input file")
if __name__=="__main__":
    input = sys.argv[1]
    output = sys.argv[2]
    frame = sys.argv[3]

    extractImages(input, output, int(frame))