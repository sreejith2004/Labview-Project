# 🎵 Sign Language Detector & LabVIEW Music Synthesizer

**A real‑time gesture‑to‑music system** combining Python-based hand‑gesture recognition with a LabVIEW front end that translates detected signs into dynamic soundscapes.

---

## 🚀 Project Overview
1. **Capture & Preprocess**  
   - Live video feed from webcam  
   - Hand landmark detection via **MediaPipe Hands**  
2. **Classification**  
   - Features extracted → **Random Forest** model classifies 15 distinct gestures  
   - Gesture label exported to gesture_output.txt  
3. **LabVIEW Integration**  
   - LabVIEW VI polls the text file  
   - Maps each gesture to musical parameters (pitch, tempo, effects)  
   - Generates adaptive audio in real time  

---

## 📦 Dependencies
- Python 3.8+  
- mediapipe  
- scikit-learn  
- opencv-python  
- LabVIEW 2021 or later  

---

## 🛠️ Installation
\\\ash
pip install mediapipe scikit-learn opencv-python
\\\
1. Clone repo  
2. Train or load \andom_forest_gesture.pkl\  
3. Open LabVIEW project \GestureMusic.vi\  

---

## ⚙️ Usage
1. Run: \python gesture_to_file.py\  
2. Launch LabVIEW VI  
3. Perform hand gestures in front of the camera → listen to responsive music  

---

## 🙌 Contributing
- Fork the repo & create feature branches  
- Submit PRs with clear descriptions  
- Report issues in GitHub Issues  

---

## 📄 License
MIT © Your Name
