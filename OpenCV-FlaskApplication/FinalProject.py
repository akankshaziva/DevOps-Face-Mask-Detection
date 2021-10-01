import numpy as np
import cv2
from tensorflow.python.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,Response
import os

application = Flask(__name__)
model = load_model('face_mask.model')
cap = cv2.VideoCapture(0)

face_cascade = \
    cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_alt2.xml'
                          )
label = {
    0: {'name': 'With mask ok', 'color': (51, 153, 255), 'id': 0},
    1: {'name': 'Mask below the nose', 'color': (255, 255, 0),
        'id': 1},
    2: {'name': 'Please wear a mask Properly', 'color': (0, 0, 255), 'id': 2},
    3: {'name': 'Please wear a mask ', 'color': (0, 102, 51), 'id': 3},
    }
def generate_frames():
    while True:
        success,frame = cap.read()  # Capture frame-by frame
        if not success:
            break
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
            color = (0, 0, 0)
            for (x, y, w, h) in faces:
                color1 = frame[y:y + h +50, x:x + w +50]
                print(color1.shape)
                if color1.shape[0] >= 200 and color1.shape[1] >= 200:
                    color1 = cv2.resize(color1, (224, 224))
                    test = image.img_to_array(color1)
                    test = np.array(test, dtype="float32")
                    gray = test / 255
                    gray = np.expand_dims(gray, axis=0)
                    print(gray.shape)
                    pred1 = gray.reshape((1, 224, 224, 3))
                    print(pred1.shape)
                    pred1 = model.predict(gray,batch_size=32)
                    pred1 = np.argmax(pred1, axis=1)  # argmax which neuron has the maximum probabilites
                    print(pred1)
                    i=pred1[-1]

                    classification = label[i]['name']
                    col = label[i]['color']

                    width = x + w
                    height = y + h
                    cv2.rectangle(frame, (x, y), (width, height), col,label[i]['id'])

                    cv2.putText(frame, classification, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
                    cv2.putText(frame, f"{len(faces)} detected face", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
                                    cv2.LINE_AA)
                    ret,buffer=cv2.imencode(".jpeg",frame)
                    frame1=buffer.tobytes()
                    yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')

@application.route('/')
def index():
    return render_template('FlaskApplication.html')
@application.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    application.run(debug=True)
    cap.release()
    cv2.destroyAllWindows()


