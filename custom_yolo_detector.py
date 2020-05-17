# *******************************************************************
#
# Author : Thanh Nguyen, 2018
# Email  : sthanhng@gmail.com
# Github : https://github.com/sthanhng
#
# BAP, AI Team
# Face detection using the YOLOv3 algorithm
#
# Description : yoloface.py
# The main code of the Face detection using the YOLOv3 algorithm
#
# *******************************************************************

# Usage example:  python yoloface.py --image samples/outside_000001.jpg \
#                                    --output-dir outputs/
#                 python yoloface.py --video samples/subway.mp4 \
#                                    --output-dir outputs/
#                 python yoloface.py --src 1 --output-dir outputs/


import argparse
import sys
import os
import time
import face_recognition
import json
import cv2
import numpy as np
from facenet import facenet
import tensorflow as tf
from imutils import face_utils
import dlib

from utils import *
from ident_utils import *

#####################################################################
args = {}
args["model_cfg"] = './cfg/yolov3-face.cfg'
args["model_weights"] = './model-weights/yolov3-wider_16000.weights'
args["src"] = "./vid.mp4"
args["output_dir"] = 'outputs/'
args["known_persons"] = json.load(open('./data/persons.json'))

#####################################################################
# print the arguments
print('----- args -----')
print('[i] The config file: ', args['model_cfg'])
print('[i] The weights of model file: ', args['model_weights'])
print('[i] Known persons in database: ', len(args['known_persons']))
print('###########################################################\n')

# check outputs directory
if not os.path.exists(args['output_dir']):
    print('==> Creating the {} directory...'.format(args['output_dir']))
    os.makedirs(args['output_dir'])
else:
    print('==> Skipping create the {} directory...'.format(args['output_dir']))

# Give the configuration and weight files for the model and load the network
# using them.
net = cv2.dnn.readNetFromDarknet(args['model_cfg'], args['model_weights'])
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

tf.Graph().as_default()
sess = tf.Session()
model = facenet.load_model("./model-weights/20180408-102900/20180408-102900.pb")
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

face_aligner = face_utils.FaceAligner(dlib.shape_predictor("./model-weights/shape_predictor_5_face_landmarks.dat"), desiredFaceWidth=160, desiredFaceHeight=160, desiredLeftEye=(0.25,0.2))


def _main():
    wind_name = 'face detection using YOLOv3'
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)

    output_file = 'result.avi'

    cap = cv2.VideoCapture(args["src"])

    # Get the video writer initialized to save the output video
    video_writer = cv2.VideoWriter(os.path.join(args['output_dir'], output_file),
                                                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                                   cap.get(cv2.CAP_PROP_FPS), (
                                                       round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                       round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    inf_time = []

    while True:
        start_time = time.time()
        has_frame, frame = cap.read()

        # Stop the program if reached end of video
        if not has_frame:
            print('[i] ==> Done processing!!!')
            print('[i] ==> Output file is stored at', os.path.join(args.output_dir, output_file))
            cv2.waitKey(1000)
            break

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

        #search for matching faces and apply names to frame
        if len(faces) > 0:
            search_identities_facenet(frame, faces, args["known_persons"], sess, embeddings, images_placeholder, phase_train_placeholder, face_aligner)        

        print('[i] ==> # detected faces: {}'.format(len(faces)))
        print('#' * 60)
        elapsed_time = time.time() - start_time
        inf_time.append(elapsed_time)

        # initialize the set of information we'll displaying on the frame
        info = [
            ('number of faces detected', '{}'.format(len(faces))),
            ('inference time in sec', '{0:.3f}'.format(elapsed_time))
        ]

        for (i, (txt, val)) in enumerate(info):
            text = '{}: {}'.format(txt, val)
            cv2.putText(frame, text, (10, (i * 20) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

        # Save the output video to file
        video_writer.write(frame.astype(np.uint8))

        cv2.imshow(wind_name, frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break

    cap.release()
    cv2.destroyAllWindows()

    print('==> All done!')
    print('***********************************************************')
    inf_time = np.array(inf_time)
    print('mean inf time', inf_time.mean())




if __name__ == '__main__':
    _main()
