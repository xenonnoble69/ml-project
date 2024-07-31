import cv2
import os
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime

# Create a directory to save face images if it doesn't exist
face_data_dir = 'face_data'
if not os.path.exists(face_data_dir):
    os.makedirs(face_data_dir)

def capture_face_data():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start capturing
    print("Press 'q' to quit and 'c' to capture a face.")
    face_id = input('Enter user ID: ')

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_img = frame[y:y+h, x:x+w]

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            img_path = os.path.join(face_data_dir, f"{face_id}.jpg")
            cv2.imwrite(img_path, face_img)
            print(f"Image saved as {img_path}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def recognize_and_mark_attendance():
    # Load known face images and create encoding list
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(face_data_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(face_data_dir, filename)
            img = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(img)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Create attendance DataFrame
    attendance_df = pd.DataFrame(columns=['Name', 'Time'])

    # Start recognizing
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            if name != "Unknown" and name not in attendance_df['Name'].values:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                attendance_df = attendance_df.append({'Name': name, 'Time': current_time}, ignore_index=True)
                attendance_df.to_csv('attendance.csv', index=False)
                print(f"Attendance marked for {name} at {current_time}")

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Select an option:")
    print("1. Capture face data")
    print("2. Recognize faces and mark attendance")
    choice = input("Enter your choice (1/2): ")

    if choice == '1':
        capture_face_data()
    elif choice == '2':
        recognize_and_mark_attendance()
    else:
        print("Invalid choice")
