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

args = {}
args["model_cfg"] = './cfg/yolov3-face.cfg'
args["model_weights"] = './model-weights/yolov3-wider_16000.weights'
args["src"] = "./vid.mp4"
args["output_dir"] = 'outputs/'
args["known_persons"] = json.load(open('./data/persons.json'))

net = cv2.dnn.readNetFromDarknet(args['model_cfg'], args['model_weights'])
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

classes = ['chewbacca', 'hux', 'leia', 'poe', 'finn', 'kylo', 'old_luke', 'rey']

face_aligner = face_utils.FaceAligner(dlib.shape_predictor("./model-weights/shape_predictor_5_face_landmarks.dat"), desiredFaceWidth=160, desiredFaceHeight=160, desiredLeftEye=(0.25,0.2))

for name in classes:
	for f in os.listdir('./data/starwars_urls/{}'.format(name)):
		if not f == '.DS_Store':
			img = cv2.imread('./data/starwars_urls/{}/{}'.format(name, f))
			print('./data/starwars_urls/{}/{}'.format(name, f))
			blob = cv2.dnn.blobFromImage(img, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)
			net.setInput(blob)
			outs = net.forward(get_outputs_names(net))
			faces = post_process(img, outs, CONF_THRESHOLD, NMS_THRESHOLD)
			if len(faces) > 0:
				for i, (left, top, width, height) in enumerate(faces):
					rect = dlib.rectangle(left,top,left+width,top+height)
					img1 = img[:,:,::-1]
					gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
					img1 = face_aligner.align(img,gray,rect)
					if i > 0:
						cv2.imwrite('./data/starwars_urls/{}/{}{}'.format(name, i, f), (img1[:,:,::-1])[:,:,::-1])
					else:
						cv2.imwrite('./data/starwars_urls/{}/{}'.format(name, f), (img1[:,:,::-1])[:,:,::-1])

















