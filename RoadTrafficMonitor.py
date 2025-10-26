import warnings
warnings.filterwarnings("ignore")

import customtkinter as ctk
from tkinter import filedialog
import cv2
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import pyttsx3
import threading
import time

from tensorflow.keras.utils import to_categorical
from keras.layers import MaxPooling2D, Dense, Flatten, Conv2D
from keras.models import Sequential

# Voice alert control
stop_voice = False

def speak_repeat(message):
    global stop_voice
    stop_voice = False

    def repeat():
        local_engine = pyttsx3.init()
        local_engine.setProperty('rate', 150)
        while not stop_voice:
            local_engine.say(message)
            local_engine.runAndWait()
            time.sleep(2)
        local_engine.stop()

    threading.Thread(target=repeat, daemon=True).start()

def stopVoiceAlert():
    global stop_voice
    stop_voice = True

# GUI Setup
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

main = ctk.CTk()
main.title("Smart Traffic Monitoring System with Emergency Detection and Voice-Based Alerts")
main.geometry("1200x750")

header = ctk.CTkLabel(main, text='Smart Traffic Monitoring System with Emergency Detection and Voice-Based Alerts',
                     font=ctk.CTkFont(size=20, weight="bold"), text_color="white", bg_color="#3b5998")
header.pack(fill="x", pady=10)

button_frame = ctk.CTkFrame(main, width=300)
button_frame.place(x=30, y=100)

output_frame = ctk.CTkFrame(main, width=800, height=550)
output_frame.place(x=350, y=100)

text = ctk.CTkTextbox(output_frame, width=780, height=530, font=("Courier", 14))
text.pack(padx=10, pady=10)

def upload():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete("0.0", "end")
    text.insert("end", filename + ' Loaded\nDataset Loaded')

def processImages():
    text.delete("0.0", "end")
    try:
        base_path = os.path.dirname(__file__)
        x_path = os.path.join(base_path, "model", "X.txt.npy")
        X_train = np.load(x_path)
        text.insert("end", f'Total images found in dataset for training = {X_train.shape[0]}\n\n')
        test = cv2.resize(X_train[30], (600, 400))
        cv2.imshow('Preprocess sample image', test)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        text.insert("end", f"❌ Error loading training images: {str(e)}\n")


def generateModel():
    global classifier
    text.delete("0.0", "end")
    try:
        base_path = os.path.dirname(__file__)
        model_weights_path = os.path.join(base_path, 'model', 'model_weights.h5')
        history_path = os.path.join(base_path, 'model', 'history.pckl')
        x_path = os.path.join(base_path, 'model', 'X.txt.npy')
        y_path = os.path.join(base_path, 'model', 'Y.txt.npy')

        if os.path.exists(model_weights_path) and os.path.exists(history_path):
            classifier = Sequential([
                Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(32, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(256, activation='relu'),
                Dense(4, activation='softmax')
            ])
            classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            classifier.load_weights(model_weights_path)

            with open(history_path, 'rb') as f:
                data = pickle.load(f)
            acc = data['accuracy'][-1] * 100
            text.insert("end", "✅ CNN Training Model Loaded Successfully\n")
            text.insert("end", f"🎯 Accuracy = {round(acc, 2)}%\n")
        else:
            X = np.load(x_path)
            Y = np.load(y_path)
            Y_cat = to_categorical(Y)

            classifier = Sequential([
                Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(32, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(256, activation='relu'),
                Dense(4, activation='softmax')
            ])
            classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            hist = classifier.fit(X, Y_cat, batch_size=16, epochs=10, shuffle=True, verbose=2)
            classifier.save_weights(model_weights_path)

            with open(history_path, 'wb') as f:
                pickle.dump(hist.history, f)

            acc = hist.history['accuracy'][-1] * 100
            text.insert("end", "✅ CNN Model Trained & Saved\n")
            text.insert("end", f"🎯 Accuracy = {round(acc, 2)}%\n")
    except Exception as e:
        text.insert("end", f"❌ Error: {str(e)}\n")

def predictTraffic():
    name = filedialog.askopenfilename(initialdir="testImages")
    if not name:
        return
    img = cv2.imread(name)
    img_array = cv2.resize(img, (64, 64)).astype('float32') / 255
    preds = classifier.predict(np.expand_dims(img_array, axis=0))
    predict_class = np.argmax(preds)
    labels = ['Accident Occurred', 'Heavy Traffic Detected', 'Fire Accident Occurred', 'Low Traffic']
    result = labels[predict_class]
    cv2.putText(img, result, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow(result, cv2.resize(img, (450, 450)))

    # Voice alerts based on detection
    if result == "Accident Occurred":
        speak_repeat("Alert! Accident occurred ahead. Drive cautiously.")
    elif result == "Fire Accident Occurred":
        speak_repeat("Warning! Fire accident reported. Evacuate area immediately.")
    elif result == "Heavy Traffic Detected":
        speak_repeat("Traffic alert. Heavy traffic detected ahead.")
    elif result == "Low Traffic":
        speak_repeat("Smooth drive. Low traffic ahead.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def graph():
    try:
        base_path = os.path.dirname(__file__)
        history_path = os.path.join(base_path, "model", "history.pckl")
        with open(history_path, 'rb') as f:
            data = pickle.load(f)

        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy / Loss')
        plt.plot(data['loss'], 'r-', label='Loss')
        plt.plot(data['accuracy'], 'g-', label='Accuracy')
        plt.legend()
        plt.title('CNN Accuracy & Loss Graph')
        plt.show()
    except Exception as e:
        text.insert("end", f"❌ Error loading graph data: {str(e)}\n")


def detectEmergencyVehicle(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bounds = [np.array([0, 100, 100]), np.array([10, 100, 100]), np.array([25, 100, 100])]
    upper_bounds = [np.array([10, 255, 255]), np.array([25, 255, 255]), np.array([35, 255, 255])]
    mask = sum([cv2.inRange(hsv, lower, upper) for lower, upper in zip(lower_bounds, upper_bounds)])
    return any(cv2.contourArea(c) > 500 for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])

def detectEmergencyVehicleButton():
    global stop_voice
    name = filedialog.askopenfilename(initialdir="testImages")
    if not name:
        return
    img = cv2.imread(name)
    if detectEmergencyVehicle(img):
        msg = 'Emergency Vehicle Detected'
        stop_voice = False
        speak_repeat("Emergency vehicle approaching. Give way immediately.")
    else:
        msg = 'No Emergency Vehicle Detected'
    cv2.putText(img, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow(msg, cv2.resize(img, (450, 450)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Buttons
ctk.CTkButton(button_frame, text="Upload Dataset", command=upload, font=("Helvetica", 14)).pack(pady=10)
ctk.CTkButton(button_frame, text="Process Dataset", command=processImages, font=("Helvetica", 14)).pack(pady=10)
ctk.CTkButton(button_frame, text="Train CNN Model", command=generateModel, font=("Helvetica", 14)).pack(pady=10)
ctk.CTkButton(button_frame, text="Predict Traffic Condition", command=predictTraffic, font=("Helvetica", 14)).pack(pady=10)
ctk.CTkButton(button_frame, text="Show Accuracy & Loss Graph", command=graph, font=("Helvetica", 14)).pack(pady=10)
ctk.CTkButton(button_frame, text="Detect Emergency Vehicle", command=detectEmergencyVehicleButton, font=("Helvetica", 14)).pack(pady=10)
ctk.CTkButton(button_frame, text="🛑 Stop Voice Alert", command=stopVoiceAlert, fg_color="red", font=("Helvetica", 14)).pack(pady=10)

main.mainloop()
