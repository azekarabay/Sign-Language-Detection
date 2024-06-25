import cv2
import mediapipe as mp
import numpy as np
import pickle

with open('hand_gesture_model.pkl', 'rb') as f:
    clf = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def classify_image(image):
    data_aux = []
    x_ = []
    y_ = []
    z_ = []

    results = hands.process(image)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z

                x_.append(x)
                y_.append(y)
                z_.append(z)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
                data_aux.append(z - min(z_))

    max_length = 126  
    if len(data_aux) < max_length:
        data_aux.extend([0] * (max_length - len(data_aux)))
    else:
        data_aux = data_aux[:max_length]

    data_aux = np.array(data_aux).reshape(1, -1)

    prediction = clf.predict(data_aux)
    predicted_class = prediction[0]

    return predicted_class

cap = cv2.VideoCapture(0)

colors = {
    'A': (255, 0, 0),   
    'B': (0, 255, 0),    
    'C': (0, 0, 255),   
    'D': (255, 255, 0),  
    'E': (255, 0, 255),  
    'F': (0, 255, 255),  
    'G': (128, 0, 0),   
    'H': (0, 128, 0),    
    'I': (0, 0, 128),   
    'J': (128, 128, 0), 
    'K': (128, 0, 128), 
    'L': (0, 128, 128), 
    'M': (192, 192, 192),
    'N': (128, 128, 128),
    'O': (255, 165, 0), 
    'P': (255, 20, 147), 
    'Q': (139, 69, 19),  
    'R': (255, 69, 0),   
    'S': (75, 0, 130),  
    'T': (240, 230, 140),
    'U': (173, 216, 230),
    'V': (220, 20, 60), 
    'W': (255, 105, 180),
    'X': (0, 191, 255), 
    'Y': (100, 149, 237),
    'Z': (32, 178, 170), 
}

while True:
    ret, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    predicted_class = classify_image(frame_rgb)

    text = 'Predicted: ' + str(predicted_class)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    text_x = 50
    text_y = 50

    cv2.rectangle(frame, (text_x, text_y - text_size[1] - 10), (text_x + text_size[0], text_y + 10), (255, 255, 255), -1)

    color = colors.get(predicted_class, (255, 255, 255))

    cv2.putText(frame, 'Predicted:', (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(frame, str(predicted_class), (text_x + 150, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

