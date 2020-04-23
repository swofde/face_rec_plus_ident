import face_recognition as fr
import numpy as np
import cv2
import json
cap = cv2.VideoCapture("./data/videos/test.mov")
data = []
wind_name = 'kek'
cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)
while True:
    has_frame, frame = cap.read()
    if not has_frame:
        print("Done!")
        break
    frame = frame[:,:,::-1]
    lctns = np.array(fr.face_locations(frame))
    if lctns.shape[0] > 0:
        lctns = lctns[0]
        frame = frame[lctns[0]:lctns[2], lctns[3]:lctns[1]]
        lndmrk = fr.face_landmarks(frame)
        embdg = fr.face_encodings(frame)
        data.append({
            "landmark" : lndmrk,
            "embeddings" : embdg
        })
        print("landmark: ", lndmrk)
        print("embedding: ", embdg)
        cv2.imshow(wind_name, frame)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break
cv2.destroyAllWndows()
cap.release()
cv2.dest
file = open("landmarks.json", 'w')
json.dump(data, file)