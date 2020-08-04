# import the necessary packages
# reference code:
# https://medium.com/datadriveninvestor/video-streaming-using-flask-and-opencv-c464bf8473d6
import cv2
import numpy as np
import pickle
from keras import backend as K
from tensorflow import Graph, Session
def join(source_dir, dest_file, read_size):
    # Create a new destination file
    output_file = open(dest_file, 'wb')

    # Get a list of the file parts
    parts = ['final_model1', 'final_model2', 'final_model3']

    # Go through each portion one by one
    for file in parts:

        # Assemble the full path to the file
        path = file

        # Open the part
        input_file = open(path, 'rb')

        while True:
            # Read all bytes of the part
            bytes = input_file.read(read_size)

            # Break out of loop if we are at end of file
            if not bytes:
                break

            # Write the bytes to the output file
            output_file.write(bytes)

        # Close the input file
        input_file.close()

    # Close the output file
    output_file.close()


join(source_dir='', dest_file="Combined_Model.p", read_size=50000000)

# defining face detector
classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
size = 4
labels_dict={0:'shaved',1:'not_shaved'}
color_dict={0:(0,255,0),1:(0,0,255)}
global loaded_model
graph1 = Graph()
with graph1.as_default():
	session1 = Session(graph=graph1)
	with session1.as_default():
		loaded_model = pickle.load(open('Combined_Model.p', 'rb'))
class VideoCamera(object):
    def __init__(self):
        # capturing video
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # releasing camera
        self.video.release()


    def get_frame(self):
        # extracting frames
        (rval, im) = self.video.read()
        im = cv2.flip(im, 1, 1)
        mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
        faces = classifier.detectMultiScale(mini)
        for f in faces:
            (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
            #Save just the rectangle faces in SubRecFaces
            face_img = im[y:y+h, x:x+w]
            resized=cv2.resize(face_img,(300,300))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,300,300,3))
            reshaped = np.vstack([reshaped])
            K.set_session(session1)
            with graph1.as_default():
                results=loaded_model.predict(reshaped)
            if results >.5:
                result = np.array([[1]])
            else:
                result = np.array([[0]])
            label = np.argmax(result)
            cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[result[label][0]],2)
            cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[result[label][0]],-1)
            cv2.putText(im, labels_dict[result[label][0]], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            # encode OpenCV raw frame to jpg and displaying it
        ret, jpeg = cv2.imencode('.jpg', im)
        return jpeg.tobytes()