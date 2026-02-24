import cv2
import os

person_name = input("Enter your name: ")
dataset_path = "dataset/"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    cv2.imshow("Capturing Images - Press q to stop", frame)

    if count % 5 == 0:
        cv2.imwrite(f"{dataset_path}{person_name}_{count}.jpg", frame)

    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q') or count > 50:
        break

cap.release()
cv2.destroyAllWindows()