from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# สร้างแอป FastAPI
app = FastAPI()

# เปิดใช้งาน CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # หรือระบุ domain ที่อนุญาต เช่น ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดโมเดลและ encoder ล่วงหน้า
model_path = "single_model_V2.pkl"
encoder_path = "menu_encoder_V2.pkl"

try:
    model = joblib.load(model_path)
    le = joblib.load(encoder_path)
except FileNotFoundError:
    model = None
    le = None

# สร้าง Pydantic Model สำหรับรับ input
class PredictionRequest(BaseModel):
    menu_name: str
    prev_day_sales: int
    day_of_week: int  # 0=จันทร์, ..., 6=อาทิตย์

# Endpoint สำหรับทำนาย
@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None or le is None:
        raise HTTPException(status_code=500, detail="❌ ไม่พบไฟล์โมเดลหรือ encoder กรุณาตรวจสอบ path ที่กำหนด")

    # ตรวจสอบค่า day_of_week
    if request.day_of_week not in range(7):
        raise HTTPException(status_code=400, detail="❌ กรุณาระบุเลขวันให้ถูกต้อง (0=จันทร์, ..., 6=อาทิตย์)")

    # ตรวจสอบว่าเมนูมีอยู่จริงหรือไม่
    try:
        menu_encoded = le.transform([request.menu_name])[0]
    except Exception:
        raise HTTPException(status_code=400, detail=f"❌ เมนู '{request.menu_name}' ยังไม่มีในระบบ")

    # เตรียมข้อมูลสำหรับทำนาย
    is_weekend = int(request.day_of_week in [5, 6])
    X_next = pd.DataFrame([{
        'day_of_week': request.day_of_week,
        'is_weekend': is_weekend,
        'prev_day_sales': request.prev_day_sales,
        'menu_encoded': menu_encoded
    }])

    # ทำนายผล
    predicted = round(model.predict(X_next)[0])
    return {
        "message": f"✅ พรุ่งนี้ควรผลิต '{request.menu_name}' ประมาณ {predicted} ชิ้น",
        "predicted_quantity": predicted
    }
