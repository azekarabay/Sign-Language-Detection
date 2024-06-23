import cv2
import mediapipe as mp
import numpy as np
import pickle

# Kaydedilen modeli yükle
with open('hand_gesture_model.pkl', 'rb') as f:
    clf = pickle.load(f)

# MediaPipe Hands modelini başlat
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def classify_image(image):
    # Görüntüyü işle ve el noktalarını çıkar
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

    # Eğitim verileriyle aynı boyuta getir
    max_length = 126  # Eğitim verilerindeki özellik sayısı
    if len(data_aux) < max_length:
        data_aux.extend([0] * (max_length - len(data_aux)))
    else:
        data_aux = data_aux[:max_length]

    data_aux = np.array(data_aux).reshape(1, -1)

    # Tahmini gerçekleştir
    prediction = clf.predict(data_aux)
    predicted_class = prediction[0]

    return predicted_class

# Kamera yakalama nesnesini oluştur
cap = cv2.VideoCapture(0)

while True:
    # Kameradan bir kare oku
    ret, frame = cap.read()

    # Kareyi BGR'den RGB'ye dönüştür
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Kareyi sınıflandır
    predicted_class = classify_image(frame_rgb)

    # Tahmin edilen sınıfı kare üzerine yazdır
    cv2.putText(frame, 'Predicted: ' + str(predicted_class), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Kareyi göster
    cv2.imshow('Hand Gesture Recognition', frame)

    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
