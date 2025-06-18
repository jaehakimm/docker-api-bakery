from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ให้ frontend จากทุก origin เรียกได้ (ใช้ตอน dev เท่านั้น ปรับเฉพาะ domain จริงตอน production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # หรือเฉพาะ domain เช่น ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load("single_model.pkl")
    le = joblib.load("menu_encoder.pkl")
except FileNotFoundError:
    print("⚠️ ไม่พบไฟล์โมเดลหรือ encoder กรุณาตรวจสอบ path และรันโค้ดส่วนเทรนโมเดลก่อน")
    model = None
    le = None

app = FastAPI()

# สร้าง Pydantic model สำหรับ Input
class PredictionInput(BaseModel):
    menu_name: str
    last_day_quantity: int
    today_day_of_week: int # 0=จันทร์, 6=อาทิตย์

@app.get("/")
def read_root():
    return {"message": "FastAPI Bakery Prediction API"}

@app.post("/predict/")
def predict_sales(input: PredictionInput):
    if model is None or le is None:
        return {"error": "Model or encoder not loaded. Please train the model first."}

    try:
        # เตรียมข้อมูลสำหรับ predict
        next_dow = (input.today_day_of_week + 1) % 7
        is_weekend = int(next_dow in [5, 6])

        # ตรวจสอบว่า menu_name อยู่ใน classes ของ encoder หรือไม่
        if input.menu_name not in le.classes_:
             return {"error": f"Menu '{input.menu_name}' not found in trained data."}

        menu_encoded = le.transform([input.menu_name])[0]

        X_next = pd.DataFrame([{
            'day_of_week': next_dow,
            'is_weekend': is_weekend,
            'prev_day_sales': input.last_day_quantity,
            'menu_encoded': menu_encoded
        }])

        # ทำนาย
        prediction = round(model.predict(X_next)[0])

        return {"predicted_quantity": max(0, int(prediction))} # ให้ค่าไม่ติดลบ

    except Exception as e:
        return {"error": str(e)}
