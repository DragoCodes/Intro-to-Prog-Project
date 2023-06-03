# Imported nececessary libraries
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import copy
# Using pre-trained model 'model_weights.h5' 
emotion_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
model = load_model('model_weights.h5')
face_casscade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')


class A_MainWindow(object):
    def image_det(self):
        self.fl = QFileDialog.getOpenFileName(filter='Image (*.*)')[0]
        if self.fl:
            frame = cv2.imread(self.fl)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_casscade.detectMultiScale(gray, 1.1, 4)
            for x, y, w, h in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                facess = face_casscade.detectMultiScale(roi_gray)
                if len(facess) == 0:
                    print("Face not detected")
                else:
                    for (ex, ey, ew, eh) in facess:
                        where = ew / 2
                        face_roi = roi_color[ey:ey+eh, ex:ex+ew]
                        gray_image = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                        resized_image = cv2.resize(gray_image, (56, 56))
                        input_data = np.expand_dims(resized_image, axis=-1)
                        input_data = input_data / 255.0
                        prediction = model.predict(np.array([input_data]))
                        text_idx = np.argmax(prediction)
                        cv2.putText(frame, emotion_list[text_idx],
                                    (x+int(where)-45, y-5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.25, (255, 0, 255), 3)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.show()

    def live_img(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            img = copy.deepcopy(frame)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_casscade.detectMultiScale(gray, 1.1, 5)
            for x, y, w, h in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                facess = face_casscade.detectMultiScale(roi_gray)
                if len(facess) == 0:
                    print("Face not detected")
                else:
                    for (ex, ey, ew, eh) in facess:
                        face_roi = roi_color[ey:ey+eh, ex:ex+ew]
                        gray_image = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                        resized_image = cv2.resize(gray_image, (56, 56))
                        input_data = np.expand_dims(resized_image, axis=-1)
                        input_data = input_data / 255.0
                        # Normalize pixel values to [0, 1]

                        # Pass the preprocessed input data
                        # to the model for prediction
                        prediction = model.predict(np.array([input_data]))
                        text_idx = np.argmax(prediction)

                        emotion_list = ['Angry', 'Disgust', 'Fear',
                                     'Happy', 'Neutral', 'Sad', 'Surprise']
                        text = emotion_list[text_idx]
                        cv2.putText(img, text, (x, y-5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45, (255, 0, 255), 2)
                        img = cv2.rectangle(img, (x, y), (x+w, y+h),
                                            (0, 0, 255), 2)
            cv2.imshow("frame", img)
            key = cv2.waitKey(1)
            if key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    def setupGui(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(770, 441)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 771, 421))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("v944-bb-16-job598.jpg"))
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.clicked.connect(self.live_img)
        self.pushButton.setGeometry(QtCore.QRect(240, 190, 121, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.clicked.connect(self.image_det)
        self.pushButton_2.setGeometry(QtCore.QRect(400, 190, 131, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.re_translateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def re_translateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Live Detection"))
        self.pushButton_2.setText(_translate("MainWindow",
                                             "Browse for Pictures"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = A_MainWindow()
    ui.setupGui(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
