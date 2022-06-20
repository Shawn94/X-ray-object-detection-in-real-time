import torch
import cv2

class Endoscope:
    def __init__(self, file, model_name, img_size):
        self.file= file
        self.model = self.load_model(model_name)
        self.img_size = img_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def load_model(self, model_name):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        return model

    def get_cords(self, show_img=False):
        """
        Gets a single image input and gives a tensor of all bbox coordinates.
        frame - take a single image
        return us coordinates of the bbox of the objects
        """
        self.model.to(self.device) #run it using cuda if available
        img = [self.file] 
        results = self.model(img)
        cords = results.xyxyn[0][:, :-1] #return tensor with xyxy coords and confidence score

        x_shape = self.img_size[0]
        y_shape = self.img_size[1]
        
        bbox_coords = list()
        for tensor in cords:
            x1, y1, x2, y2 = int(tensor[0]*x_shape)-30, int(tensor[1]*y_shape)-30, int(tensor[2]*x_shape)+30, int(tensor[3]*y_shape)+30     
            bbox = [x1,y1,x2,y2]
            bbox_coords.append(bbox)

        if show_img==True: self.plot_img(bbox_coords)

        return bbox_coords

    def plot_img(self, cords):
            frame = cv2.imread(self.file)
            bgr = (0, 255, 0)

            for i in cords:
                cv2.rectangle(frame, (i[0],i[1]), (i[2], i[3]), bgr, 1)
            cv2.imshow('Prove of bbox', frame)
            cv2.waitKey(0)
            # Destroying present windows on screen
            cv2.destroyAllWindows()


object = Endoscope(capture_frame='cheetah_dataset/test/images/IMG_0028_jpeg.rf.514f114438ecc7b4598469edfa2ab9d5.jpg',
                model_name='saved_models/cheetah_yolov5s.pt', img_size=(640,640)
)

print(object.get_cords())
