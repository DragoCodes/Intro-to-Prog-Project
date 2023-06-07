'''
This Module is for the Detector
'''
# Imported nececessary libraries
import copy
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import numpy as np
from cv2 import CascadeClassifier
from cv2 import imread
from cv2 import cvtColor
from cv2.data import haarcascades
from cv2 import COLOR_BGR2GRAY
from cv2 import resize
from cv2 import rectangle
from cv2 import putText
from cv2 import FONT_HERSHEY_SIMPLEX
from cv2 import COLOR_BGR2RGB
from cv2 import VideoCapture
from cv2 import waitKey
from cv2 import imshow
from cv2 import destroyAllWindows
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
# Using pre-trained model 'model_weights.h5'
model = load_model('model_weights.h5')
emotion_list = ['Angry', 'Disgust', 'Fear', 'Happy',
                'Neutral', 'Sad', 'Surprise']
face_casscade = CascadeClassifier(
  haarcascades + 'haarcascade_frontalface_default.xml')


class A_MainWindow(object):
    def __init__(self):
        """Initialization"""
        self.centralwidget = None
        self.label = None
        self.label_2 = None
        self.push_button = None
        self.push_button_2 = None
        self.statusbar = None
        self.fil = None

    def image_det(self):
        """For Emotion detection in Pictures"""
        self.fil = QFileDialog.getOpenFileName(filter='Image (*.*)')[0]
        if self.fil:
            frame = imread(self.fil)
            gray = cvtColor(frame, COLOR_BGR2GRAY)
            faces = face_casscade.detectMultiScale(gray, 1.1, 4)
            for x_coordinate, y_coordinate, width, height in faces:
                roi_gray = gray[y_coordinate:y_coordinate+height,
                                x_coordinate:x_coordinate+width]
                roi_color = frame[y_coordinate:y_coordinate+height,
                                  x_coordinate:x_coordinate+width]
                rectangle(frame, (x_coordinate, y_coordinate),
                          (x_coordinate+width, y_coordinate+height),
                          (255, 0, 0), 2)
                facess = face_casscade.detectMultiScale(roi_gray)
                if len(facess) == 0:
                    print("Face not detected")
                else:
                    for (ex_coordinate,
                         ey_coordinate,
                         ewidth,
                         eheight) in facess:
                        where = ewidth / 2
                        face_roi = roi_color[
                            ey_coordinate:ey_coordinate+eheight,
                            ex_coordinate:ex_coordinate+ewidth
                            ]
                        gray_image = cvtColor(face_roi, COLOR_BGR2GRAY)
                        resized_image = resize(gray_image, (56, 56))
                        input_data = np.expand_dims(resized_image, axis=-1)
                        input_data = input_data / 255.0
                        prediction = model.predict(np.array([input_data]))
                        text_idx = np.argmax(prediction)
                        putText(frame, emotion_list[text_idx],
                                (x_coordinate+int(where)-45,
                                 y_coordinate-5),
                                FONT_HERSHEY_SIMPLEX, 1.25,
                                (255, 0, 255), 3)
            plt.imshow(cvtColor(frame, COLOR_BGR2RGB))
            plt.show()

    def live_img(self):
        """ For live capture"""
        cap = VideoCapture(0)
        while True:
            ret, frame = cap.read()
            img = copy.deepcopy(frame)
            gray = cvtColor(img, COLOR_BGR2GRAY)
            faces = face_casscade.detectMultiScale(gray, 1.1, 5)
            for x_coordinate, y_coordinate, width, height in faces:
                roi_gray = gray[y_coordinate:y_coordinate+height,
                                x_coordinate:x_coordinate+width]
                roi_color = frame[y_coordinate:y_coordinate+height,
                                  x_coordinate:x_coordinate+width]
                rectangle(frame, (x_coordinate, y_coordinate),
                          (x_coordinate+width, y_coordinate+height),
                          (255, 0, 0), 2)
                facess = face_casscade.detectMultiScale(roi_gray)
                if len(facess) == 0:
                    print("Face not detected")
                else:
                    for (ex_coordinate,
                         ey_coordinate,
                         ewidth,
                         eheight) in facess:
                        face_roi = roi_color[
                            ey_coordinate:ey_coordinate+eheight,
                            ex_coordinate:ex_coordinate+ewidth
                            ]
                        gray_image = cvtColor(face_roi, COLOR_BGR2GRAY)
                        resized_image = resize(gray_image, (56, 56))
                        input_data = np.expand_dims(resized_image, axis=-1)
                        input_data = input_data / 255.0
                        # Normalize pixel values to [0, 1]

                        # Pass the preprocessed input data
                        # to the model for prediction
                        prediction = model.predict(np.array([input_data]))
                        text_idx = np.argmax(prediction)

                        emotion_list = [
                          'Angry', 'Disgust', 'Fear', 'Happy',
                          'Neutral', 'Sad', 'Surprise']
                        text = emotion_list[text_idx]
                        putText(img, text, (x_coordinate, y_coordinate-5),
                                FONT_HERSHEY_SIMPLEX,
                                0.45, (255, 0, 255), 2)
                        img = rectangle(img, (x_coordinate, y_coordinate),
                                        (x_coordinate+width,
                                         y_coordinate+height),
                                        (0, 0, 255), 2)
            imshow("frame", img)
            key = waitKey(1)
            if key == 27:
                break
        cap.release()
        destroyAllWindows()

    def setupGui(self, main_window):
        """This function setups the main window"""
        main_window.setObjectName("MainWindow")
        main_window.resize(770, 441)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 771, 421))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("v944-bb-16-job598.jpg"))
        self.label.setObjectName("label")
        self.push_button = QtWidgets.QPushButton(self.centralwidget)
        self.push_button.clicked.connect(self.live_img)
        self.push_button.setGeometry(QtCore.QRect(240, 190, 121, 41))
        self.push_button.setObjectName("pushButton")
        self.push_button_2 = QtWidgets.QPushButton(self.centralwidget)
        self.push_button_2.clicked.connect(self.image_det)
        self.push_button_2.setGeometry(QtCore.QRect(400, 190, 131, 41))
        self.push_button_2.setObjectName("pushButton_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(260, 30, 251, 61))
        font = QtGui.QFont()
        font.setFamily("Segoe Print")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        main_window.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        main_window.setStatusBar(self.statusbar)

        self.retranslate_ui(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslate_ui(self, main_window):
        """Retranslate the user interface elements to the main window"""
        _translate = QtCore.QCoreApplication.translate
        main_window.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.push_button.setText(_translate("MainWindow", "Live Detection"))
        self.push_button_2.setText(_translate("MainWindow",
                                              "Browse for Pictures"))
        self.label_2.setText(_translate("MainWindow", "FaceDontLie"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = A_MainWindow()
    ui.setupGui(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
