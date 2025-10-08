# 🌍 REALM — Renewable Energy Automated Land Monitoring

**REALM (Renewable Energy Automated Land Monitoring)** is a student-built AI web application that classifies land based on satellite imagery to estimate its suitability for renewable or non-renewable energy projects.

Users can:
- Enter **latitude and longitude** to fetch satellite and weather data, or  
- **Upload an image** directly for AI-based prediction.

---

## ⚙️ Tech Stack
- **Backend:** Flask (Python)
- **Model:** ResNet (PyTorch)
- **Frontend:** HTML, CSS, JavaScript
- **APIs:** NASA WMS & Open-Meteo

---

## 🧠 About the Model
Our current model is trained on a **small, low-resolution dataset** sourced from **Kaggle** and manually labeled satellite images.  
It is **not trained on high-end or government datasets**, so accuracy is limited — this is an **early-stage proof of concept**.  

In future iterations, we aim to:
- Use **high-resolution imagery** (NASA, Sentinel, PlanetScope)  
- Retrain with larger and more diverse datasets  
- Improve accuracy and scalability for real-world use  

---

## 🚀 Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Ashmit-rai17/REALM.git
cd REALM
2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate     # (Windows: venv\Scripts\activate)
3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
python app.py


Then open your browser and visit:

http://127.0.0.1:5000/

📂 Project Structure
REALM/
├── app.py              # Flask backend (model + API routes)
├── templates/
│   └── index.html      # Web interface
├── model/              # Trained model + class names
├── uploads/            # User-uploaded images
├── requirements.txt    # Dependencies
└── README.md

👨‍💻 Team

Ashmit Rai — AI Model Development

Aditya Raj — Frontend & Backend Integration

Divya Bhatia — Data Preparation & Preprocessing

📈 Future Plans

Expand dataset and retrain model on high-resolution satellite images

Add map-based coordinate selection

Deploy on Render / Hugging Face Spaces

Build a mobile version for real-time field usage

⚠️ REALM is an academic project — not yet optimized for commercial or field-grade accuracy.
We aim to improve it over time with better datasets and model fine-tuning.