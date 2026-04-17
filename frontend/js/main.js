let currentDiagnosisId = null;
let currentPrediction = null;

async function uploadImage() {
    const file = document.getElementById("imageInput").files[0];
    if (!file) {
        alert("Please upload an image first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    formData.append("patient_id", document.getElementById("patientId").value);
    formData.append("patient_name", document.getElementById("patientName").value);
    formData.append("age", document.getElementById("age").value);
    formData.append("gender", document.getElementById("gender").value);

    const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    // Show prediction
    document.getElementById("ceap").innerText = data.ceap;
    document.getElementById("severity").innerText = data.severity;
    document.getElementById("confidence").innerText = data.confidence;
    document.getElementById("treatment").innerText = data.treatment;
    document.getElementById("xaiText").innerText = data.xai_text || "";
    document.getElementById("note").innerText = data.note || "";

    // Show images
    document.getElementById("heatmapImage").src =
        "http://127.0.0.1:8000" + data.heatmap_url;

    document.getElementById("originalImage").src =
        URL.createObjectURL(file);

    currentDiagnosisId = data.diagnosis_id || null;
    currentPrediction = data.ceap;
}

async function submitFeedback() {
    if (!currentPrediction) {
        alert("Run a prediction first.");
        return;
    }

    const feedbackData = {
        diagnosis_id: currentDiagnosisId,
        prediction: currentPrediction,
        user_feedback: document.getElementById("feedbackStatus").value,
        correct_label: document.getElementById("correctLabel").value,
        remarks: document.getElementById("remarks").value
    };

    await fetch("http://127.0.0.1:8000/feedback", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(feedbackData)
    });

    alert("Feedback submitted successfully!");
}
