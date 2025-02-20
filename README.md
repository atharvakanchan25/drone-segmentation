Drone Image Segmentation using DeepLabV3+
🚀 Semantic segmentation of aerial drone images using DeepLabV3+ with ResNet-50.
This project trains and evaluates a model for segmenting drone images into 20 different classes.

![DeepLab Prediction](https://github.com/user-attachments/assets/e9106779-34cb-4252-8f08-655cc13d60e2)


📦 drone-segmentation
├── dataset/                         # Contains original images & masks
├── train.py                          # Main script for training & evaluation
├── requirements.txt                   # Required Python libraries
├── README.md                          # Project documentation
└── models/
    ├── deeplabv3_drone.pth            # Trained model weights


📋 Dataset Information
Source: Semantic Drone Dataset
Classes (20):
tree, grass, dirt, gravel, rocks, water, paved area, pool, person, dog, car, bicycle, roof, wall, fence, window, door, obstacle

⚙️ Installation
1️⃣ Clone the Repository

git clone https://github.com/atharvakanchan25/drone-segmentation.git
cd drone-segmentation

2️⃣ Set Up Virtual Environment (Recommended)

python3 -m venv drone_env
source drone_env/bin/activate  # On Linux/Mac
drone_env\\Scripts\\activate   # On Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

🚀 Training & Evaluation
1️⃣ Train the Model (DeepLabV3+)

python3 train.py

The model will be trained using CrossEntropyLoss & AdamW optimizer.
Trained weights will be saved as models/deeplabv3_drone.pth.
2️⃣ Evaluate the Model

python3 train.py

Mean IoU & Pixel Accuracy will be displayed.
Prediction visualizations will be plotted.

📊 Visualizations
✅ Loss Curve & IoU Trends
✅ Per-Class IoU Bar Chart
✅ Confusion Matrix
✅ Sample Predictions vs. Ground Truth

🛠 Future Improvements
 Run inference on new images
 Compare with other models (UNet, HRNet, SegFormer)
 Optimize inference with ONNX / TorchScript
📝 License
This project is for academic & research purposes only, following the Semantic Drone Dataset license.


