# 📌 Base Image
FROM python:3.10-slim

# 🛠 Set working directory
WORKDIR /app

# 🧾 Copy all source code and model files
COPY . .

# 🐍 Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn joblib pandas scikit-learn

# 🌐 Expose FastAPI port
EXPOSE 8000

# 🚀 Start the app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
