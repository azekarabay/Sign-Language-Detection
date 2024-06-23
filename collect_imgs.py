import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Alfabe için sınıf sayısını belirleyin
number_of_classes = 26  # 26 harf

dataset_size = 100  # Her sınıf için veri sayısı

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    # Harfler için sınıf adı
    class_name = chr(ord('A') + j)

    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {class_name}')

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Ready for class {class_name}? Press "Q" to start!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.putText(frame, f'Capturing image for class {class_name}: {counter+1}/{dataset_size}. Press "Q" to capture!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        key = cv2.waitKey(25)
        if key == ord('q'):
            cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
            counter += 1
        elif key == ord('e'):
            break

cap.release()
cv2.destroyAllWindows()