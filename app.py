"""import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from mtcnn import MTCNN

app = Flask(__name__, template_folder='face_recognition/templates' )

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = MTCNN()

# Global model loading
model = None
model_path = 'static/face_recognition_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)

def get_camera():
    for i in range(2):  # Try camera index 0 and 1
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
        else:
            cap.release()
    return None

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time\n')

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    if img is not None and img.size != 0:
        faces = face_detector.detect_faces(img)
        face_points = []
        for face in faces:
            x, y, w, h = face['box']
            face_points.append((x, y, w, h))
        return face_points
    else:
        return []

def identify_face(facearray):
    global model
    if model is None:
        return None
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, model_path)
    global model
    model = knn  # Update global model

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    return df['Name'], df['Roll'], df['Time'], len(df)

def add_attendance(name):
    if not name:
        return
    parts = name.split('_')
    if len(parts) < 2:
        return
    username = parts[0]
    userid = parts[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if str(userid) not in df['Roll'].astype(str).values:
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    global model
    if model is None:
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2, mess='No trained model found in static folder. Please add a new face first.')

    cap = get_camera()
    if cap is None:
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2, mess='Camera not found or cannot be opened.')

    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        faces = extract_faces(frame)
        if faces:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))
            if isinstance(identified_person,np.ndarray) and identified_person.size > 0:
                add_attendance(identified_person[0])
                cv2.putText(frame, f'{identified_person[0]}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        newusername = request.form.get('newusername')
        newuserid = request.form.get('newuserid')

        # Validation
        if not newusername or not newuserid:
            names, rolls, times, l = extract_attendance()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Username or Roll ID cannot be empty.')

        userimagefolder = os.path.join('static', 'faces', f'{newusername}_{newuserid}')
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)

        cap = get_camera()
        if cap is None:
            names, rolls, times, l = extract_attendance()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Camera not found or cannot be opened.')

        i, j = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 10 == 0 and i < 50:
                    name = f'{newusername}_{i}.jpg'
                    cv2.imwrite(os.path.join(userimagefolder, name), frame[y:y + h, x:x + w])
                    i += 1
                j += 1
            cv2.imshow('Adding new User', frame)
            if j == 500 or cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        train_model()
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)
    else:
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

if __name__ == '__main__':
    app.run(debug=True)

    





import cv2
import os
import logging
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from mtcnn import MTCNN
import pyttsx3

app = Flask(__name__, template_folder='face_recognition/templates')

# Setup Logging
logging.basicConfig(level=logging.INFO)

# Text-to-Speech engine
engine = pyttsx3.init()

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Global variables
face_detector = MTCNN()
model = None
model_path = 'models/face_recognition_model.pkl'

# Load model if exists
if os.path.exists(model_path):
    model = joblib.load(model_path)
    logging.info("Model loaded successfully.")

# Ensure directories exist
for folder in ['Attendance', 'static', 'static/faces', 'models']:
    os.makedirs(folder, exist_ok=True)

# Initialize attendance file
attendance_file = f'Attendance/Attendance-{datetoday}.csv'
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Time\n')

def get_camera():
    for i in range(2):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
        else:
            cap.release()
    return None

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    if img is not None and img.size != 0:
        faces = face_detector.detect_faces(img)
        return [(f['box'][0], f['box'][1], f['box'][2], f['box'][3]) for f in faces]
    return []

def identify_face(facearray):
    global model
    if model is None:
        return None
    return model.predict(facearray)

def train_model():
    faces, labels = [], []
    for user in os.listdir('static/faces'):
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (100, 100))
            faces.append(resized_face.ravel())
            labels.append(user)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(np.array(faces), labels)
    joblib.dump(knn, model_path)
    global model
    model = knn
    logging.info("Model trained and saved successfully.")

def extract_attendance():
    df = pd.read_csv(attendance_file)
    return df['Name'], df['Roll'], df['Time'], len(df)

def add_attendance(name):
    if not name: return
    parts = name.split('_')
    if len(parts) < 2: return
    username, userid = parts[0], parts[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(attendance_file)
    if str(userid) not in df['Roll'].astype(str).values:
        with open(attendance_file, 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')
        engine.say(f"Attendance marked for {username}")
        engine.runAndWait()

@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    global model
    if model is None:
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2, mess='No trained model found. Please add a new face.')

    cap = get_camera()
    if cap is None:
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2, mess='Camera not found or cannot be opened.')

    ret = True
    face_found = False
    while ret:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        faces = extract_faces(frame)
        if faces:
            face_found = True
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))
            if isinstance(identified_person, np.ndarray) and identified_person.size > 0:
                add_attendance(identified_person[0])
                current_time = datetime.now().strftime("%H:%M:%S")
                cv2.putText(frame, f'{identified_person[0]} - {current_time}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
        else:
            cv2.putText(frame, 'No face detected. Please adjust your position.', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=("Attendance marked successfully" if face_found else "No face detected."))

@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        newusername = request.form.get('newusername')
        newuserid = request.form.get('newuserid')
        if not newusername or not newuserid:
            return render_template('home.html', **extract_attendance(), totalreg=totalreg(), datetoday2=datetoday2, mess='Username or Roll ID cannot be empty.')

        userimagefolder = os.path.join('static', 'faces', f'{newusername}_{newuserid}')
        os.makedirs(userimagefolder, exist_ok=True)

        cap = get_camera()
        if cap is None:
            return render_template('home.html', **extract_attendance(), totalreg=totalreg(), datetoday2=datetoday2, mess='Camera not found or cannot be opened.')

        i, j = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
                if j % 10 == 0 and i < 50:
                    name = f'{newusername}_{i}.jpg'
                    cv2.imwrite(os.path.join(userimagefolder, name), frame[y:y + h, x:x + w])
                    i += 1
                j += 1
            cv2.imshow('Adding new User', frame)
            if j == 500 or cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        train_model()

    return render_template('home.html', **extract_attendance(), totalreg=totalreg(), datetoday2=datetoday2, mess="New face added successfully.")

if __name__ == '__main__':
    app.run(debug=True)"""


import os
import cv2
import logging
import numpy as np
import pandas as pd
import joblib
import pyttsx3
from datetime import date, datetime
from flask import Flask, request, render_template
from mtcnn import MTCNN
from sklearn.neighbors import KNeighborsClassifier

# Initialize Flask app
app = Flask(__name__, template_folder='face_recognition/templates')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('attendance_system.log'),
        logging.StreamHandler()
    ]
)

# Constants
DATE_FORMAT = "%m_%d_%y"
DATE_FORMAT_DISPLAY = "%d-%B-%Y"
MODEL_PATH = 'models/face_recognition_model.pkl'
IMAGE_SIZE = (100, 100)  # Consistent size for training and recognition
SAMPLES_PER_USER = 50
FACE_DETECTION_INTERVAL = 10  # Process every 10th frame during enrollment

# Initialize components with error handling
try:
    face_detector = MTCNN()
except Exception as e:
    logging.warning(f"MTCNN initialization failed: {e}. Falling back to Haar Cascade.")
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
except Exception as e:
    logging.warning(f"TTS engine initialization failed: {e}")
    engine = None

# Ensure directories exist
for folder in ['Attendance', 'static/faces', 'models']:
    os.makedirs(folder, exist_ok=True)

# Initialize model
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")

# Helper functions
def get_current_date_files():
    today = date.today()
    datetoday = today.strftime(DATE_FORMAT)
    datetoday2 = today.strftime(DATE_FORMAT_DISPLAY)
    attendance_file = f'Attendance/Attendance-{datetoday}.csv'
    return datetoday, datetoday2, attendance_file

def initialize_attendance_file():
    _, _, attendance_file = get_current_date_files()
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write('Name,Roll,Time\n')

def get_camera():
    for i in range(3):  # Try up to 3 cameras
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            logging.info(f"Camera {i} opened successfully.")
            return cap
        cap.release()
    logging.error("No available cameras found.")
    return None

def extract_faces(img):
    try:
        if isinstance(face_detector, MTCNN):
            faces = face_detector.detect_faces(img)
            return [(f['box'][0], f['box'][1], f['box'][2], f['box'][3]) for f in faces]
        else:  # Haar Cascade
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            return faces
    except Exception as e:
        logging.error(f"Face detection failed: {e}")
        return []

def preprocess_face(face_img):
    try:
        resized = cv2.resize(face_img, IMAGE_SIZE)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return gray.ravel()
    except Exception as e:
        logging.error(f"Face preprocessing failed: {e}")
        return None

def train_model():
    global model
    faces, labels = [], []
    
    try:
        for user in os.listdir('static/faces'):
            user_dir = os.path.join('static/faces', user)
            if not os.path.isdir(user_dir):
                continue
                
            for imgname in os.listdir(user_dir):
                img_path = os.path.join(user_dir, imgname)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                face_features = preprocess_face(img)
                if face_features is not None:
                    faces.append(face_features)
                    labels.append(user)
        
        if len(faces) == 0:
            logging.warning("No faces found for training.")
            return False
            
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(np.array(faces), labels)
        
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(knn, MODEL_PATH)
        model = knn
        logging.info(f"Model trained with {len(faces)} samples from {len(set(labels))} users.")
        return True
        
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        return False

def extract_attendance():
    _, _, attendance_file = get_current_date_files()
    try:
        df = pd.read_csv(attendance_file)
        return df['Name'].tolist(), df['Roll'].tolist(), df['Time'].tolist(), len(df)
    except Exception as e:
        logging.error(f"Error reading attendance file: {e}")
        return [], [], [], 0

def add_attendance(name):
    if not name:
        return False
        
    try:
        parts = name.split('_')
        if len(parts) < 2:
            return False
            
        username, userid = parts[0], parts[1]
        current_time = datetime.now().strftime("%H:%M:%S")
        _, _, attendance_file = get_current_date_files()
        
        df = pd.read_csv(attendance_file)
        if str(userid) not in df['Roll'].astype(str).values:
            with open(attendance_file, 'a') as f:
                f.write(f'\n{username},{userid},{current_time}')
            
            if engine:
                engine.say(f"Attendance marked for {username}")
                engine.runAndWait()
                
            logging.info(f"Attendance recorded for {username} ({userid}) at {current_time}")
            return True
        return False
    except Exception as e:
        logging.error(f"Error adding attendance: {e}")
        return False

# Routes
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html',
                         names=names,
                         rolls=rolls,
                         times=times,
                         l=l,
                         totalreg=len(os.listdir('static/faces')),
                         datetoday2=get_current_date_files()[1])

@app.route('/start', methods=['GET'])
def start_attendance():
    if model is None:
        return render_template('home.html',
                            **get_template_data(),
                            mess='No trained model found. Please add users first.')

    cap = get_camera()
    if cap is None:
        return render_template('home.html',
                            **get_template_data(),
                            mess='Camera not available.')

    face_found = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = extract_faces(frame)
            if faces:
                face_found = True
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                face_img = frame[y:y+h, x:x+w]
                face_processed = preprocess_face(face_img)
                
                if face_processed is not None:
                    identity = model.predict([face_processed])[0]
                    if add_attendance(identity):
                        current_time = datetime.now().strftime("%H:%M:%S")
                        cv2.putText(frame, f'{identity} - {current_time}',
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Position your face in the frame',
                          (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                          1, (0, 0, 255), 2)

            cv2.imshow('Attendance System', frame)
            if cv2.waitKey(1) == 27:  # ESC key
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return render_template('home.html',
                         **get_template_data(),
                         mess='Attendance completed' if face_found else 'No face detected')

@app.route('/add', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        newusername = request.form.get('newusername', '').strip()
        newuserid = request.form.get('newuserid', '').strip()

        if not newusername or not newuserid:
            return render_template('home.html',
                                **get_template_data(),
                                mess='Both username and ID are required.')

        if not newuserid.isdigit():
            return render_template('home.html',
                                **get_template_data(),
                                mess='ID must be numeric.')

        userfolder = os.path.join('static', 'faces', f'{newusername}_{newuserid}')
        os.makedirs(userfolder, exist_ok=True)

        cap = get_camera()
        if cap is None:
            return render_template('home.html',
                                **get_template_data(),
                                mess='Camera not available.')

        samples_collected = 0
        frame_count = 0
        
        try:
            while samples_collected < SAMPLES_PER_USER:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % FACE_DETECTION_INTERVAL != 0:
                    continue

                faces = extract_faces(frame)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    status = f"Collected {samples_collected}/{SAMPLES_PER_USER} samples"
                    cv2.putText(frame, status, (30, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    face_img = frame[y:y+h, x:x+w]
                    img_path = os.path.join(userfolder, f'{newusername}_{samples_collected}.jpg')
                    cv2.imwrite(img_path, face_img)
                    samples_collected += 1
                    break

                cv2.imshow('Enrolling New User', frame)
                if cv2.waitKey(1) == 27:  # ESC key
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        if samples_collected > 0:
            if train_model():
                message = f"Successfully added {newusername} with {samples_collected} samples"
            else:
                message = "User added but model training failed"
        else:
            message = "Failed to capture face samples"

        return render_template('home.html',
                            **get_template_data(),
                            mess=message)

    return render_template('home.html', **get_template_data())

def get_template_data():
    names, rolls, times, l = extract_attendance()
    return {
        'names': names,
        'rolls': rolls,
        'times': times,
        'l': l,
        'totalreg': len(os.listdir('static/faces')),
        'datetoday2': get_current_date_files()[1]
    }

if __name__ == '__main__':
    initialize_attendance_file()
    app.run(host='0.0.0.0', port=5000, debug=True)