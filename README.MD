# 🦺 PPEv2 – Real-Time PPE Detection System (YOLOv8)

PPEv2 is a computer vision-based safety system that detects Personal Protective Equipment (PPE) such as helmets and IFR suits in real time using YOLOv8.  
The system supports both **model training** and **live webcam detection**, with automatic logging of results for analysis.

---

## 🚀 Features

- 🧠 YOLOv8-based object detection
- 🎥 Real-time webcam PPE detection
- ⚡ GPU acceleration (CUDA support)
- 📊 Automatic logging to CSV
- 🏷️ Bounding box + label visualization
- 🔍 Adjustable confidence & IoU thresholds
- 🏗️ End-to-end pipeline (training → inference)

---

## 📂 Project Structure
ppev2/
│── README.dataset.txt # Dataset information (Roboflow export)
│── README.roboflow.txt # Roboflow metadata
│── data.yaml # YOLOv8 dataset configuration
│── detections.csv # Output detection log
│── train_ppe.py # Model training script
│── webcam.py # Real-time detection script
│── README.md

---

## 📊 Dataset

- **Source:** Roboflow PPE Detection Dataset  
- **Images:** 464  
- **Format:** YOLOv8  
- **Classes:** Helmet, IFR Suit, PPE Violations  
- **Resolution:** 640×640  

🔗 Dataset Link:  
https://universe.roboflow.com/manab-jyoti-goswami-lqvan/ppe-detection-ykpqj  

### Preprocessing Applied
- Auto-orientation (EXIF removed)
- Resize to 640×640 (stretch)
- No augmentation

---

## ⚙️ Installation

```bash
git clone https://github.com/jatinpochiraju/ppev2.git
cd ppev2

 Train Model
python train_ppe.py

Output:
runs/detect/train/weights/best.pt
 
Run Detection
python webcam.py
Uses webcam (VIDEO_SRC = 0)
Press ESC to stop
