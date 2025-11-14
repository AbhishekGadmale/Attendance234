import os
import base64
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
from datetime import datetime, date
import numpy as np
import cv2
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib

app = Flask(__name__, template_folder='templates', static_folder='static')

# Config
nimgs = 10
attendance_dir = 'Attendance'
attendance_file = os.path.join(attendance_dir, 'Attendance.csv')
faces_root = os.path.join('static', 'faces')
model_path = os.path.join('static', 'face_recognition_model.pkl')
haar_path = 'haarcascade_frontalface_default.xml'

# Ensure directories and files exist
os.makedirs(attendance_dir, exist_ok=True)
os.makedirs(faces_root, exist_ok=True)
if not os.path.isfile(attendance_file):
    df = pd.DataFrame(columns=['Name','Roll','Time','Date','Class'])
    df.to_csv(attendance_file, index=False)

# Helper functions
face_detector = cv2.CascadeClassifier(haar_path)

def totalreg():
    return len([d for d in os.listdir(faces_root) if os.path.isdir(os.path.join(faces_root, d))])


def extract_faces(img):
    if img is None or img.size == 0:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
    return faces


def train_model():
    X, y = [], []
    users = [d for d in os.listdir(faces_root) if os.path.isdir(os.path.join(faces_root, d))]
    for user in users:
        user_folder = os.path.join(faces_root, user)
        for fname in os.listdir(user_folder):
            fpath = os.path.join(user_folder, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            face = cv2.resize(img, (50,50))
            X.append(face.ravel())
            y.append(user)
    if len(X) == 0:
        return False
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(np.array(X), np.array(y))
    joblib.dump(knn, model_path)
    return True


def identify_face(facearray):
    if not os.path.isfile(model_path):
        return None
    model = joblib.load(model_path)
    return model.predict(facearray)


def add_attendance(name, selected_class):
    try:
        username, userid = name.split('_')
    except Exception:
        username = name
        userid = ''
    current_time = datetime.now().strftime('%H:%M:%S')
    current_date = date.today().isoformat()
    df = pd.read_csv(attendance_file)
    # Ensure Class column exists
    if 'Class' not in df.columns:
        df['Class'] = ''
    # Prevent duplicate for same date & class
    try:
        userid_int = int(userid) if userid != '' else userid
    except Exception:
        userid_int = userid
    mask = (df['Roll'] == userid_int) & (df['Date'] == current_date) & (df['Class'] == selected_class)
    if not mask.any():
        df.loc[len(df.index)] = [username, userid_int, current_time, current_date, selected_class]
        df.to_csv(attendance_file, index=False)
    return True

# ROUTES
@app.route('/')
def dashboard():
    df = pd.read_csv(attendance_file)
    if 'Class' not in df.columns:
        df['Class'] = ''
    names = df['Name'].tolist()
    rolls = df['Roll'].tolist()
    times = df['Time'].tolist()
    dates = df['Date'].tolist()
    classes = df['Class'].tolist()
    l = len(df)
    attendance_counts = df['Class'].value_counts().to_dict()
    return render_template('dashboard.html', names=names, rolls=rolls, times=times, dates=dates, classes=classes, l=l, totalreg=totalreg(), attendance_counts=attendance_counts)


@app.route('/add_user_api', methods=['POST'])
def add_user_api():
    data = request.get_json()
    username = data.get('username')
    userid = data.get('userid')
    image_b64 = data.get('image')
    if not username or not userid or not image_b64:
        return jsonify({'status':'error','msg':'Missing data'}), 400

    folder = os.path.join(faces_root, f"{username}_{userid}")
    os.makedirs(folder, exist_ok=True)

    # Save image
    try:
        header, encoded = image_b64.split(',',1)
    except ValueError:
        encoded = image_b64
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'status':'error','msg':'Invalid image'}), 400

    count = len([f for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    fname = os.path.join(folder, f"{count}.jpg")
    cv2.imwrite(fname, img)

    # Train model automatically when enough images
    if count+1 >= nimgs:
        train_model()

    return jsonify({'status':'success','saved': count+1})


@app.route('/recognize_api', methods=['POST'])
def recognize_api():
    data = request.get_json()
    image_b64 = data.get('image')
    selected_class = data.get('class','')
    if not image_b64:
        return jsonify({'found':False})
    try:
        header, encoded = image_b64.split(',',1)
    except ValueError:
        encoded = image_b64
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    faces = extract_faces(frame)
    if len(faces) == 0:
        return jsonify({'found':False})
    (x,y,w,h) = faces[0]
    face = cv2.resize(frame[y:y+h, x:x+w], (50,50))
    result = identify_face(face.reshape(1,-1))
    if result is None or len(result)==0:
        return jsonify({'found':False})
    person = result[0]
    add_attendance(person, selected_class)
    return jsonify({'found':True,'name':person})


@app.route('/export_class', methods=['GET'])
def export_class():
    selected_class = request.args.get('class')
    df = pd.read_csv(attendance_file)
    if 'Class' not in df.columns:
        df['Class'] = ''
    if selected_class:
        filtered = df[df['Class'] == selected_class]
    else:
        filtered = df
    outpath = os.path.join(attendance_dir, f"Attendance_{selected_class or 'all'}.csv")
    filtered.to_csv(outpath, index=False)
    return send_file(outpath, as_attachment=True)


@app.route('/record/<int:roll>', methods=['GET'])
def view_record(roll):
    df = pd.read_csv(attendance_file)
    if 'Class' not in df.columns:
        df['Class'] = ''
    user_data = df[df['Roll']==roll]
    return render_template('view_record.html', user_data=user_data.to_dict('records'))


@app.route('/clear', methods=['POST'])
def clear_attendance():
    df = pd.DataFrame(columns=['Name','Roll','Time','Date','Class'])
    df.to_csv(attendance_file, index=False)
    return redirect(url_for('dashboard'))


# Run
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
