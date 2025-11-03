# 🖼️ CIFAR-10 Image Classifier (FastAPI + CNN)

[![Live Demo](https://img.shields.io/badge/Live_Demo-Visit_App-brightgreen?style=for-the-badge)](https://dl-assignment-2205412-application.onrender.com)
[![Built with FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Powered by PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Deployed on Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)

This is a web application that classifies uploaded images into one of the 10 classes from the CIFAR-10 dataset. The backend is built with **FastAPI** and serves a **Convolutional Neural Network (CNN)** model trained from scratch in PyTorch.

This project was originally part of a university assignment (`DL_Assignment_2205412`) to compare a CNN against a Vision Transformer (ViT). This deployed version uses only the more efficient and accurate CNN model to work within the 512MB memory limits of Render's free hosting tier.

## 🚀 Live Demo

**You can try the live app here:**
**[https://dl-assignment-2205412-application.onrender.com](https://dl-assignment-2205412-application.onrender.com)**

*(Note: The app is on Render's free tier, so it may take 30-60 seconds to "wake up" if it hasn't been used recently.)*

## ℹ️ Note on Versions

The **live deployed version** on Render uses only the **CNN model**. This was done to reduce memory usage and stay within the free tier's 512MB RAM limit.

If you want to run the **complete version** that includes both the **CNN and ViT classifiers**, please use the code from the first commit:

* **Commit:** `1f2fd74`
* **Link:** [https://github.com/SChakraborty04/DL_Assignment_2205412_Application/commit/1f2fd7406a53885c6aa3114380f092580da2399b](https://github.com/SChakraborty04/DL_Assignment_2205412_Application/commit/1f2fd7406a53885c6aa3114380f092580da2399b)

---

## 🛠️ Technology Stack

* **Backend:** FastAPI, Uvicorn, Gunicorn
* **Machine Learning:** PyTorch
* **Image Processing:** Pillow (PIL)
* **Frontend:** Simple HTML, CSS, and vanilla JavaScript
* **Deployment:** Render
* **Package Management:** `uv` / `pip`
* **Version Control:** Git & Git LFS (for handling `.pth` model files)

---

## 🤖 Model Details

The classifier is a **SimpleCNN** model trained for 12 epochs on the CIFAR-10 dataset.

* **Best Test Accuracy:** **79.92%**
* **Classes:** `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

### Model Architecture
The model is a simple 4-layer CNN with the following structure:
* `Conv2d (3, 32)` -> `BatchNorm` -> `ReLU` -> `MaxPool`
* `Conv2d (32, 64)` -> `BatchNorm` -> `ReLU` -> `MaxPool`
* `Conv2d (64, 128)` -> `BatchNorm` -> `ReLU` -> `MaxPool`
* `Conv2d (128, 256)` -> `BatchNorm` -> `ReLU` -> `MaxPool`
* `Linear (256*2*2, 512)` -> `ReLU` -> `Dropout(0.3)`
* `Linear (512, 10)`

---

## ⚙️ How to Run Locally

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/SChakraborty04/DL_Assignment_2205412_Application.git](https://github.com/SChakraborty04/DL_Assignment_2205412_Application.git)
    cd DL_Assignment_2205412_Application
    ```
    *(To run the dual-model version, check out the specific commit mentioned above after cloning)*
    ```bash
    git checkout 1f2fd7406a53885c6aa3114380f092580da2399b
    ```

2.  **Pull Model Files (if using Git LFS):**
    This project uses Git LFS for the `.pth` model files. Make sure you have it installed.
    ```bash
    git lfs install
    git lfs pull
    ```
    *If you don't use Git LFS, just manually place your `best_simple_cnn.pth` (and `best_vision_transformer.pth` for the first commit) in the root directory.*

3.  **Create and Activate a Virtual Environment (using `uv`):**
    ```bash
    # Install uv if you don't have it (e.g., pip install uv)
    uv venv
    source .venv/bin/activate  # macOS/Linux
    # .\.venv\Scripts\Activate.ps1  # Windows PowerShell
    ```

4.  **Install Dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

5.  **Run the Server:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```

6.  **Open the App:**
    Go to `http://127.0.0.1:8000` in your browser.

---

## ☁️ Deployment on Render

This app is configured for deployment on Render.

* **Runtime:** `Python 3`
* **Build Command:** `pip install -r requirements.txt`
* **Start Command:** `gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app`