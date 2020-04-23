import numpy as np
import cv2
import face_recognition
from utils import refined_box

CONF_THRESHOLD = 0.6
CONF_THRESHOLD_FACENET = 1.0

COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)

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
	for (left, top, width, height) in boxes:
		crops.append(frame[top:top+height, left:left+width,::-1])
	#for each crop compute embedding vectors
	for c in crops:
		cropped_embeddings.append(face_recognition.face_encodings(c))
	#for each embedding
	for e in cropped_embeddings:
		#if not null or not of shape (0,) 
		if e is not None and len(e) > 0:
			#find argmin and minimum distance to current embedding from known embeddings
			amin, vmin = _findMin(e, persons_info)
			if vmin <= CONF_THRESHOLD:
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
		width = box[2]
		height = box[3]
		(left, top, right, bottom) = refined_box(left, top, width, height)
		draw_names(frame, names[i], left, top, right, bottom)


def _findMin(target, persons_info):
	ans = [np.linalg.norm(_ - target) for _ in np.array([_["embeddings"] for _ in persons_info])]
	amin = np.argmin(ans)
	return amin, ans[amin] 

def search_identities_facenet(frame, boxes, persons_info, facenet_session, embeddings, images_placeholder, phase_train_placeholder):
	crops = []
	names = []
	cropped_embeddings = []

	#crop faces from frame using predicted bboxes
	for (left, top, width, height) in boxes:
		crops.append(cv2.resize(frame[top:top+height, left:left+width,::-1], (160,160), interpolation=cv2.INTER_LINEAR))
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
		width = box[2]
		height = box[3]
		(left, top, right, bottom) = refined_box(left, top, width, height)
		draw_names(frame, names[i], left, top, right, bottom)