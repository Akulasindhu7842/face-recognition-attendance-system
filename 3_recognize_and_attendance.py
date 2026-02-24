import cv2
import numpy as np
import pandas as pd
from datetime import datetime

# Load trained model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("trained_model.yml")

# Load labels
labels = np.load("labels.npy", allow_pickle=True).item()

# Create attendance file if not exists
attendance_file = "attendance.csv"
try:
    pd.read_csv(attendance_file)
except:
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(attendance_file, index=False)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]

        label, confidence = model.predict(face)

        if confidence < 90:
            name = labels[label]

            # Show name above face
            cv2.putText(frame, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Mark attendance
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            df = pd.read_csv(attendance_file)

            if not ((df["Name"] == name) & (df["Date"] == date)).any():
                df.loc[len(df)] = [name, date, time]
                df.to_csv(attendance_file, index=False)
                print(f"Attendance marked for: {name}")

        else:
            cv2.putText(frame, "Unknown", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()