import torch
import numpy as np
import cv2
from time import time
import yaml

from saved_models.models.yolo import Model


class Endoscope:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def __init__(self, capture_index, model_name, data='./dataset/cheetah_dataset.yaml'):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        if data:  # data.yaml path (optional)
            with open(data, errors='ignore') as f:
                self.classes = yaml.safe_load(f)['names']  # class names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """

        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        #model = torch.load(model_name)
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        #self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape)-30, int(row[1]*y_shape)-30, int(row[2]*x_shape)+30, int(row[3]*y_shape)+30
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 1)

        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = self.get_video_capture()
        assert cap.isOpened()

        #create video file to save output frames
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter('example.mp4',fourcc, int(fps),(640,640))
        
        frame_counter = 0
        while True:

            ret, frame = cap.read()
            assert ret
            
            vid_frame = cv2.resize(frame, (640,640))
            
            start_time = time()
            results = self.score_frame(vid_frame)
            frame = self.plot_boxes(results, vid_frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
            
            #If the last frame is reached, reset the capture and the frame_counter
            frame_counter+=1
            if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                frame_counter = 0 #Or whatever as long as it is the same as next line
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                output.release() #stop writing frames to output video when it was played for one loop

            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            cv2.imshow('Cheetah Detection', frame)
            output.write(frame) #write every frame to output video
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        
        cap.release()
        cv2.destroyAllWindows()

        

        
        
# Create a new object and execute.
detector = Endoscope(capture_index='dataset/cheetah_vid.mp4', model_name='./saved_models/cheetah_yolov5s.pt')
detector()