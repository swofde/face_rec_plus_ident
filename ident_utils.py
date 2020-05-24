import cv2
import dlib
import numpy as np
import face_recognition
import matplotlib.pyplot as plt
from utils import refined_box
from imutils import face_utils

CONF_THRESHOLD_IDENT = 0.6
CONF_THRESHOLD_FACENET = 0.7

COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)

IS_GRADIENT_BOOST = True

def draw_names(frame, name, left, top, right, bottom):
	frame_height = frame.shape[0]
	frame_width = frame.shape[1]
	text = name
	# Display the label at the top of the bounding box
	label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
	top = max(top, label_size[1])
	cv2.putText(frame, text, (left + 40, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GREEN, 1)


def search_identities(frame, boxes, persons_info):
	crops = []
	names = []
	cropped_embeddings = []

	#crop faces from frame using predicted bboxes
	"""for (left, top, width, height) in boxes:
					crops.append(frame[top:top+height, left:left+width,::-1])"""

	for (left, top, right, bottom) in boxes:
		crops.append(frame[top:bottom, left:right])
	#for each crop compute embedding vectors
	for c in crops:
		cropped_embeddings.append(face_recognition.face_encodings(c))
	#for each embedding
	for e in cropped_embeddings:
		#if not null or not of shape (0,) 
		if e is not None and len(e) > 0:
			#find argmin and minimum distance to current embedding from known embeddings
			amin, vmin = _findMin(e, persons_info)
			if vmin <= CONF_THRESHOLD_IDENT:
				print("found person ", persons_info[amin]["name"]) 
				names.append(persons_info[amin]["name"])
			else:
				names.append("unknown")
		else:
			names.append("unknown")
	
	for i in range(len(names)):
		box = boxes[i]
		left = box[0]
		top = box[1]
		right = box[2]
		bottom = box[3]
		(left, top, right, bottom) = (left, top , right, bottom)
		draw_names(frame, names[i], left, top, right, bottom)

def search_identities_facenet(frame, boxes, persons_info, facenet_session, embeddings, images_placeholder, phase_train_placeholder, face_aligner):
	crops = []
	names = []
	cropped_embeddings = []

	#align and crop faces from frame using predicted bboxes
	for (left, top, right, bottom) in boxes:
		rect = dlib.rectangle(left,top,right,bottom)
		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		crops.append(face_aligner.align(frame,gray,rect))

	#for each crop compute embedding vectors
	crops = np.array(crops).astype(np.uint8)
	feed_dict = {images_placeholder: crops, phase_train_placeholder: False}
	cropped_embeddings = facenet_session.run(embeddings, feed_dict=feed_dict)
	#for each embedding
	for e in cropped_embeddings:
		#if not null or not of shape (0,) 
		if e is not None and len(e) > 0:
			#find argmin and minimum distance to current embedding from known embeddings
			amin, vmin = _findMin(e, persons_info)
			if vmin <= CONF_THRESHOLD_FACENET:
				print("found person ", persons_info[amin]["name"]) 
				names.append(persons_info[amin]["name"])
			else:
				names.append("unknown")
		else:
			names.append("unknown")

	for i in range(len(names)):
		box = boxes[i]
		left = box[0]
		top = box[1]
		right = box[2]
		bottom = box[3]
		draw_names(frame, names[i], left, top, right, bottom)

def search_identities_facenet_boostbased(frame, boxes, classnames, classifier, facenet_session, embeddings, images_placeholder, phase_train_placeholder, face_aligner):
	crops = []
	names = []
	cropped_embeddings = []

	#align and crop faces from frame using predicted bboxes
	for (left, top, right, bottom) in boxes:
		rect = dlib.rectangle(left,top,right,bottom)
		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		crops.append(face_aligner.align(frame,gray,rect))

	#for each crop compute embedding vectors
	crops = np.array(crops).astype(np.uint8)
	feed_dict = {images_placeholder: crops, phase_train_placeholder: False}
	cropped_embeddings = facenet_session.run(embeddings, feed_dict=feed_dict)

	#for each embedding
	for e in cropped_embeddings:
		#if not null or not of shape (0,) 
		if e is not None and len(e) > 0:
			#find argmin and minimum distance to current embedding from known embeddings
			prob, amax = predict_gradient_boost(classifier, [e])
			print("found person {} with probability of {}".format(classnames[amax], prob*100))
			names.append("{}, prob: '{}".format(classnames[amax], round(prob, 2)))
		else:
			names.append("unknown")

	for i in range(len(names)):
		box = boxes[i]
		left = box[0]
		top = box[1]
		right = box[2]
		bottom = box[3]
		draw_names(frame, names[i], left, top, right, bottom)

def _findMin(target, persons_info):
	ans = [np.linalg.norm(_ - target) for _ in np.array([_["embeddings"] for _ in persons_info])]
	amin = np.argmin(ans)
	print(ans)
	return amin, ans[amin] 
def predict_gradient_boost(model, target):
	predicted = model.predict_proba(target)
	amax = np.argmax(predicted, axis=1)[0]
	print(amax)
	return predicted[0, amax], amax













