# 

import pickle
import numpy as np
import cv2
import warnings
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import matplotlib.pyplot as plt


# Supressão de warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)

app = FastAPI()

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: int

# Variável global para o modelo carregado
xgb_model_carregado = None

# Função para carregar o modelo salvo
def load_model():
    global xgb_model_carregado
    with open("xgb_model.pkl", "rb") as f:
        xgb_model_carregado = pickle.load(f)

# Função chamada ao iniciar o servidor
@app.on_event("startup")
async def startup_event():
    load_model()

# Endpoint de predição
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Leitura da imagem
    img = await file.read()
    np_img = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)

    # Redimensionar a imagem para 8x8, que é o formato esperado pelo modelo
    img_resized = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)

    # Mostrar a imagem (apenas para debug, opcional)
    plt.imshow(img_resized, cmap='gray')
    plt.title('Imagem Carregada')
    plt.show()

    # Normalizar os valores da imagem para ficar entre 0 e 9
    img_resized = (9 * (img_resized / 255.0)).astype(int)

    # Achatar a imagem para que seja um vetor de 64 elementos, como o modelo espera
    img_array = img_resized.flatten().reshape(1, -1)

    # Fazer a predição
    prediction = xgb_model_carregado.predict(img_array)

    return {"prediction": int(prediction[0])}


# Endpoint de healthcheck
@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}
