# Sign Language Gesture Recognition System 🤟

A real-time sign language recognition system that uses **MediaPipe** hand landmark detection and a trained **Keras deep learning model** to recognize hand gestures via webcam.

---

## 📌 Project Overview

This system detects hand landmarks in real-time using your webcam and classifies them into one of 6 predefined sign language gestures. The recognized gesture is displayed live on screen.

**Recognized Gestures:**
| Gesture | Meaning |
|---------|---------|
| `hello` | Hello |
| `thank_you` | Thank You |
| `mom` | Mom |
| `dada` | Dada |
| `me` | Me |
| `tanu` | Tanu |

---

## 🗂️ Project Structure

```
Sign-Language-Gesture-Recognition/
│
├── src/
│   └── main.py                        # Real-time gesture recognition via webcam
│
├── model/
│   └── gesture_recognition_model.h5   # Trained Keras model
│
├── data/
│   ├── data.npy                        # Combined training data
│   ├── labels.npy                      # Training labels
│   ├── hello.npy                       # Gesture data — Hello
│   ├── thank_you.npy                   # Gesture data — Thank You
│   ├── mom.npy                         # Gesture data — Mom
│   ├── dada.npy                        # Gesture data — Dada
│   ├── me.npy                          # Gesture data — Me
│   └── tanu.npy                        # Gesture data — Tanu
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-org>/Sign-Language-Gesture-Recognition.git
cd Sign-Language-Gesture-Recognition
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
python src/main.py
```

- Allow webcam access when prompted
- Show your hand gesture in front of the camera
- The recognized gesture label appears on screen in **green text**
- Press **`Q`** to quit

---

## 🧠 How It Works

1. **Webcam captures** each frame in real-time
2. **MediaPipe Hands** detects 21 hand landmarks (x, y coordinates)
3. Landmarks are passed to the **Keras model** for classification
4. The predicted **gesture label** is displayed on the frame
5. Hand connections drawn with green dots and red lines for visual feedback

---

## 🛠️ Tech Stack

| Tool | Details |
|------|---------|
| Python | 3.8.9 |
| OpenCV | Real-time video capture |
| MediaPipe | Hand landmark detection |
| TensorFlow / Keras | Gesture classification model |
| NumPy | Data handling |
| Platform | PyCharm |

---

## 👩‍💻 Author

**Tanuja Devi. M**
---

## 📄 License

This project is for educational purposes.
