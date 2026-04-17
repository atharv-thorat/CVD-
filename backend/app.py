from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import io, torch, uuid, cv2, numpy as np
import os
from typing import Optional

from backend.predictor import predict_image
from backend.database import Database
from backend.gradcam import GradCAM
from backend.model.model_architecture import ImprovedAttentionCNN


app = FastAPI(title="CVD AI Diagnostic API")

# ---------------------- STATIC DIRECTORIES ----------------------
os.makedirs("static/heatmaps", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
# ---------------------- CORS ----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- DATABASE ----------------------
@app.on_event("startup")
async def startup_db():
    await Database.connect_db()

@app.on_event("shutdown")
async def shutdown_db():
    await Database.close_db()

# ---------------------- MODEL LOADING ----------------------
MODEL_PATH = "backend/model/final_improved_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = ImprovedAttentionCNN(num_classes=5, pretrained=False)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.to(DEVICE)
model.eval()

target_layer = model.layer4[-1].conv2
gradcam = GradCAM(model, target_layer)

# ---------------------- Pydantic Models ----------------------
class PatientInfo(BaseModel):
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None


class Feedback(BaseModel):
    diagnosis_id: Optional[str]
    prediction: str
    user_feedback: str
    correct_label: Optional[str] = None
    remarks: Optional[str] = None


# ---------------------- XAI TEXT GENERATOR ----------------------
def generate_xai_text(ceap, confidence):
    explanations = {
        "C0": "The model observed normal vascular texture and uniform pixel distribution indicating no visible abnormality.",
        "C1": "Slight vessel dilation patterns were detected, indicating early stage venous changes.",
        "C2_C3": "Strong activation over dilated and irregular vein regions suggesting moderate venous insufficiency.",
        "C4": "Heatmap shows concentration around skin texture variations indicating pigmentation or skin damage.",
        "C5_C6": "High attention on ulcerated and inflamed areas, strongly suggesting severe venous disease."
    }

    base_text = explanations.get(ceap, "Prediction based on learned vascular and texture features.")
    return f"{base_text} Model confidence is {round(confidence*100,2)}%."


# ---------------------- HOME ----------------------
@app.get("/")
def home():
    return {
        "message": "CVD AI Diagnostic System",
        "version": "1.0",
        "device": DEVICE,
        "status": "online"
    }

# ---------------------- PREDICTION API ----------------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    patient_id: Optional[str] = None,
    patient_name: Optional[str] = None,
    age: Optional[int] = None,
    gender: Optional[str] = None
):
    try:
        # Read image
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")

        # Model Prediction
        ceap, severity, confidence, class_idx, input_tensor = predict_image(model, image_pil)

        # GradCAM
        cam = gradcam.generate(input_tensor, class_idx)

        # Overlay heatmap
        img = np.array(image_pil.resize((224, 224)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        # Save heatmap
        filename = f"{uuid.uuid4().hex}.png"
        path = f"static/heatmaps/{filename}"
        cv2.imwrite(path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))

        # Clinical note
        note = None
        if confidence < 0.6:
            note = "⚠️ Low confidence. Specialist review recommended."

        treatment = get_treatment(ceap)

        # XAI Explanation
        xai_text = generate_xai_text(ceap, confidence)

        # Response
        result = {
            "ceap": ceap,
            "severity": severity,
            "confidence": round(confidence, 3),
            "heatmap_url": f"/static/heatmaps/{filename}",
            "treatment": treatment,
            "xai_text": xai_text,
            "note": note
        }

        # Save Diagnosis
        if patient_id or patient_name:
            diagnosis_data = {
                "patient_id": patient_id,
                "patient_name": patient_name,
                "age": age,
                "gender": gender,
                "diagnosis": result,
                "image_filename": file.filename,
                "heatmap_path": path
            }

            doc_id = await Database.save_diagnosis(diagnosis_data)
            result["diagnosis_id"] = doc_id
            result["saved_to_db"] = True

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------- FEEDBACK API ----------------------
@app.post("/feedback")
async def submit_feedback(feedback: Feedback):
    try:
        await Database.save_feedback(feedback.dict())
        return {"status": "Feedback saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------- HISTORY APIs ----------------------
@app.get("/patient/{patient_id}/history")
async def get_patient_history(patient_id: str):
    try:
        history = await Database.get_patient_history(patient_id)
        return {"patient_id": patient_id, "total_diagnoses": len(history), "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/diagnosis/{diagnosis_id}")
async def get_diagnosis(diagnosis_id: str):
    try:
        diagnosis = await Database.get_diagnosis_by_id(diagnosis_id)
        if not diagnosis:
            raise HTTPException(status_code=404, detail="Diagnosis not found")

        diagnosis["_id"] = str(diagnosis["_id"])
        return diagnosis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------- TREATMENT LOGIC ----------------------
def get_treatment(ceap):
    treatments = {
        "C0": "✅ Normal: Regular exercise, stay hydrated, elevate legs when resting",
        "C1": "Compression stockings, avoid prolonged standing",
        "C2_C3": "⚠️ Doppler ultrasound recommended, compression therapy, vascular consultation",
        "C4": "⚠️ Medication for skin changes, wound care, vascular surgery consult",
        "C5_C6": "🚨 URGENT: Active ulcer treatment, vascular surgery required immediately"
    }
    return treatments.get(ceap, "Consult healthcare provider")
