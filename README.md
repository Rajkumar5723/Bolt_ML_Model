# 🔩 Bolt Tightness Detection

This project is a **bolt tightness detection system** using **ResNet50V2** and **Streamlit** for an interactive web-based interface. The application takes an image of a bolt and classifies it as **Tight** or **Loose** based on a trained deep learning model.

## 🚀 Features
- **Deep Learning Model**: Uses ResNet50V2 for image classification.
- **GPU Support**: Automatically detects and utilizes a GPU if available.
- **Interactive Web Interface**: Built with Streamlit for easy image upload and classification.
- **Confidence Score**: Provides a confidence level for each prediction.

## 📂 Project Structure
```
├── app.py                # Main Streamlit app
├── bolt_tightness.py     # Secondary script with similar functionality
├── Model.ipynb           # Jupyter Notebook for training and evaluation
├── bolt_tightness_model_resnet.h5  # Trained model file (Ensure you have it in the same directory)
├── requirements.txt      # List of dependencies
├── README.md             # Project documentation
```

## 🛠 Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2️⃣ Install Dependencies
Create a virtual environment (optional but recommended):
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
Install required packages:
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```sh
streamlit run app.py
```
This will launch the web app in your browser.

## 🎯 Usage
1. Upload an image of a bolt (JPG/PNG format).
2. The model analyzes the image and classifies it as **Tight** or **Loose**.
3. A confidence score is displayed along with the result.

## 🏗 Model Training
The deep learning model is trained using ResNet50V2. To retrain the model, refer to `Model.ipynb` and ensure you have the dataset properly structured.

## 🤝 Contributing
Feel free to fork this repository, submit issues, or open pull requests to enhance functionality!

## 📜 License
This project is licensed under the **MIT License**.

---

💡 *Developed with TensorFlow & Streamlit for real-time bolt classification!*
