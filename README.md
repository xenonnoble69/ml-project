# Face Recognition and Attendance System

This project implements a face recognition and attendance system using OpenCV and face_recognition libraries in Python. The system captures face data using a webcam or pictures, recognizes faces, and stores their names along with a timestamp in a CSV file for attendance tracking.

## Features

- Capture face data from a webcam and save it as images.
- Recognize faces from live video feed.
- Mark attendance by storing recognized faces' names and timestamps in a CSV file.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/xenonnoble69/face-recognition-attendance.git
    cd face-recognition-attendance
    ```

2. Install the required libraries:
    ```bash
    pip install opencv-python numpy pandas face_recognition
    ```

## Usage

### Step 1: Capture and Store Face Data

Run the `capture_faces.py` script to capture face data from your webcam and save it as image files:
```bash
python capture_faces.py
