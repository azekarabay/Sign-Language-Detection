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

# Harfler için renkler
colors = {
    'A': (255, 0, 0),    # Kırmızı
    'B': (0, 255, 0),    # Yeşil
    'C': (0, 0, 255),    # Mavi
    'D': (255, 255, 0),  # Sarı
    'E': (255, 0, 255),  # Mor
    'F': (0, 255, 255),  # Camgöbeği
    'G': (128, 0, 0),    # Koyu Kırmızı
    'H': (0, 128, 0),    # Koyu Yeşil
    'I': (0, 0, 128),    # Koyu Mavi
    'J': (128, 128, 0),  # Zeytin Yeşili
    'K': (128, 0, 128),  # Eflatun
    'L': (0, 128, 128),  # Teal
    'M': (192, 192, 192),# Gümüş
    'N': (128, 128, 128),# Gri
    'O': (255, 165, 0),  # Turuncu
    'P': (255, 20, 147), # Pembe
    'Q': (139, 69, 19),  # Kahverengi
    'R': (255, 69, 0),   # Koyu Turuncu
    'S': (75, 0, 130),   # Çivit Mavisi
    'T': (240, 230, 140),# Haki
    'U': (173, 216, 230),# Açık Mavi
    'V': (220, 20, 60),  # Krem
    'W': (255, 105, 180),# Sıcak Pembe
    'X': (0, 191, 255),  # Derin Gökyüzü Mavisi
    'Y': (100, 149, 237),# Mısır Çiçeği Mavisi
    'Z': (32, 178, 170), # Açık Deniz Mavisi
}

while True:
    # Kameradan bir kare oku
    ret, frame = cap.read()

    # Kareyi BGR'den RGB'ye dönüştür
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Kareyi sınıflandır
    predicted_class = classify_image(frame_rgb)

    # Tahmin edilen sınıfı kare üzerine yazdır
    text = 'Predicted: ' + str(predicted_class)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    # Metnin boyutunu al
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Metnin konumunu belirle
    text_x = 50
    text_y = 50

    # Arka plan dikdörtgeni çiz (beyaz)
    cv2.rectangle(frame, (text_x, text_y - text_size[1] - 10), (text_x + text_size[0], text_y + 10), (255, 255, 255), -1)

    # Tahmin edilen sınıfa göre rengi belirle
    color = colors.get(predicted_class, (255, 255, 255))

    # "Predicted" metnini siyah yaz
    cv2.putText(frame, 'Predicted:', (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    # Harfi yaz
    cv2.putText(frame, str(predicted_class), (text_x + 150, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    # Kareyi göster
    cv2.imshow('Hand Gesture Recognition', frame)

    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()

