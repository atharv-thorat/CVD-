# CVD (Cardiovascular Disease) Prediction Project

A deep learning-based system for cardiovascular disease detection and diagnosis using image analysis with explainability through Grad-CAM visualizations.

## Project Overview

This project implements a neural network model to detect and classify cardiovascular disease from medical images. It includes:
- **Image preprocessing and patch-based analysis** with multiple patch sizes (15, 25, 50)
- **Deep learning model** for disease classification
- **Grad-CAM visualization** for explainable AI insights
- **REST API backend** built with FastAPI
- **Web-based frontend** for user interaction

## Project Structure

```
CVD_Project/
├── backend/                      # Flask/FastAPI backend application
│   ├── app.py                   # Main application entry point
│   ├── database.py              # Database operations
│   ├── predictor.py             # Model prediction logic
│   ├── evaluate.py              # Model evaluation metrics
│   ├── gradcam.py               # Grad-CAM explanation generation
│   ├── model_architecture.py    # Neural network architecture
│   └── model/
│       └── final_improved_model.pth  # Pre-trained model weights
│
├── frontend/                     # Web interface
│   ├── index.html               # Main HTML page
│   ├── js/
│   │   └── main.js              # Frontend logic
│   └── css/
│       └── style.css            # Styling
│
├── CVI-img-datasets/            # Dataset directory
│   ├── imagedata/               # Raw images organized by class (1-5)
│   ├── imagePatch-labels15/     # 15x15 patch labels
│   ├── imagePatch-labels25/     # 25x25 patch labels
│   ├── imagePatch-labels50/     # 50x50 patch labels
│   ├── processed_data/          # Pre-processed training/validation/test splits
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── MODEL_CARD.md            # Model documentation
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- CUDA 11.8+ (recommended for GPU acceleration)

### Setup Instructions

1. **Clone or download the project**
```bash
cd CVD_Project
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download pre-trained model** (if not included)
   - Place `final_improved_model.pth` in `backend/model/`

## Usage

### Running the Backend Server

```bash
cd backend
python app.py
```

The API will be available at `http://localhost:8000`

API endpoints:
- `POST /predict` - Get prediction for an uploaded image
- `POST /predict-with-heatmap` - Get prediction with Grad-CAM visualization
- `GET /health` - Health check

### Running the Frontend

Open `frontend/index.html` in a web browser or serve it with a local server:
```bash
cd frontend
python -m http.server 8080
```

Then navigate to `http://localhost:8080`

## Model Details

- **Architecture**: Deep Convolutional Neural Network
- **Input**: Medical images (RGB or grayscale)
- **Output**: Disease classification (5 classes) with confidence scores
- **Training Data**: CVI image dataset with patch-based labels
- **Explainability**: Grad-CAM for visual attention maps

## Dataset Organization

- **imagedata/1-5/**: Original images organized by disease class
- **imagePatch-labels15/25/50/**: Patch-based annotations at different resolutions
- **processed_data/**: Train/validation/test splits ready for model training

## Features

✅ Multi-scale patch analysis (15×15, 25×25, 50×50)
✅ Explainable predictions with Grad-CAM heatmaps
✅ RESTful API for easy integration
✅ Web-based user interface
✅ Model evaluation metrics
✅ Batch processing support

## Dependencies

Key packages:
- **PyTorch** - Deep learning framework
- **FastAPI** - Web framework
- **OpenCV** - Image processing
- **Pillow** - Image manipulation
- **NumPy** - Numerical computing
- **PyMongo** - Database (optional)

See `requirements.txt` for complete list.

## Performance Metrics

Results are evaluated using:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC scores

See `CVI-img-datasets/training_results.json` for detailed metrics.

## Model Card

For detailed model information, licensing, and ethical considerations, see [MODEL_CARD.md](CVI-img-datasets/MODEL_CARD.md)

## Contributing

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## License

[Add your license information here]

## Authors

[Add author/contributor information here]

## Support

For issues or questions:
1. Check existing documentation
2. Review model card and training results
3. Check API logs in `backend/results.txt`

---

**Last Updated**: April 2026
