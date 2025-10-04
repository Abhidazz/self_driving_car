# Self Driving Car 

##  Overview

This project implements a **self-driving car using deep learning**.
The car learns to steer by imitating human driving behavior, using data collected from a simulator.
The solution is based on the **NVIDIA End-to-End Learning for Self-Driving Cars** model architecture.

##  Features

* **Data Collection**: Images (`center`, `left`, `right`) and driving logs (`driving_log.csv`).
* **Data Augmentation**: Random pan, zoom, brightness changes, and horizontal flips.
* **Preprocessing Pipeline**: Cropping, resizing, Gaussian blur, YUV color conversion, normalization.
* **Model Architecture**: Deep CNN inspired by **NVIDIA architecture** with ELU activations and dropout.
* **Training**:

  * Loss: Mean Squared Error (MSE)
  * Optimizer: Adam (learning rate `1e-3`)
  * Batch generator with real-time augmentation
* **Inference**:

  * Flask + SocketIO server communicates with the simulator.
  * Model predicts steering angles in real-time.
  * Throttle is adjusted dynamically to maintain safe speed.

##  Project Structure

```
.
├── data/                      # Dataset (driving_log.csv + IMG folder)
│   ├── driving_log.csv
│   └── IMG/
├── model/                     # Saved trained model
│   └── model.h5
├── model_training.ipynb          # Training script
├── drive.py                   # Flask + SocketIO server to run model in simulator
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🔧 Installation

### 1. Clone repository

```bash
git clone (https://github.com/Abhidazz/self_driving_car.git)
cd self-driving-car
```

### 2. Create environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

##  Training the Model

```bash
python model_training.py
```

This will:

* Load and preprocess the dataset
* Train the CNN model
* Save the final model in `model/model.h5`

##  Running the Car in Simulator

1. Launch the **Udacity Self-Driving Car Simulator** in Autonomous mode.

2. Run the server:

   python drive.py
    
   If not connected,!!
   pip install python-engineio==3.13.2
   pip install python-socketio==4.6.1

3. The car will drive using the trained model 🚘

##  Results

* Training Loss: Decreases steadily with augmentation.
* Validation Loss: Stable with early stopping & dropout.
* Car drives smoothly around the track without going off-road.

##  Troubleshooting

* **FileNotFoundError**: Ensure all images in `driving_log.csv` exist inside `IMG/`.
* **np.sctypes error**: Downgrade NumPy (`pip install numpy==1.26.4`).
* **mse error on model load**: Use `model = load_model("model/model.h5", compile=False)`.
* **Permission issues on Windows**: Run terminal as Administrator.

##  References

* [NVIDIA End-to-End Learning for Self-Driving Cars](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)
* [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim)

---
