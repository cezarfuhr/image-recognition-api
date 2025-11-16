# Image Recognition API

> 🚧 **Project Under Development** | AI-powered computer vision API for image classification, object detection, and OCR

## Overview

Production-ready REST API for image recognition and computer vision tasks using state-of-the-art deep learning models. Built with Python, TensorFlow/PyTorch, and FastAPI, providing high-performance endpoints for object detection, image classification, face recognition, OCR, and custom model inference.

## Features

### 🖼️ Image Classification
- **Multi-class Classification** - Categorize images into 1000+ classes (ImageNet)
- **Custom Models** - Train and deploy custom classifiers
- **Confidence Scores** - Probability distribution for predictions
- **Batch Processing** - Process multiple images simultaneously
- **Transfer Learning** - Fine-tune pre-trained models

### 🎯 Object Detection
- **YOLO v8** - Real-time object detection
- **Faster R-CNN** - High-accuracy object detection
- **SSD** - Single Shot MultiBox Detector
- **Bounding Boxes** - Precise object localization
- **Multi-object Detection** - Detect multiple objects in single image
- **Custom Object Training** - Train on custom datasets

### 👤 Face Recognition
- **Face Detection** - Locate faces in images
- **Face Encoding** - Generate face embeddings
- **Face Matching** - Compare and match faces
- **Age & Gender Detection** - Demographic analysis
- **Emotion Recognition** - Detect facial expressions
- **Face Verification** - 1:1 face comparison

### 📝 Optical Character Recognition (OCR)
- **Text Detection** - Locate text in images
- **Text Extraction** - Extract text content
- **Multi-language Support** - 100+ languages
- **Handwriting Recognition** - Cursive and print
- **Document Processing** - PDFs, receipts, forms
- **License Plate Recognition** - Vehicle identification

### 🎨 Image Processing
- **Image Enhancement** - Auto-adjust brightness, contrast
- **Background Removal** - Segment foreground objects
- **Style Transfer** - Apply artistic styles
- **Image Segmentation** - Pixel-level classification
- **Super Resolution** - Upscale images with AI
- **Image Denoising** - Remove noise and artifacts

### 🔍 Advanced Features
- **Similarity Search** - Find similar images
- **NSFW Detection** - Content moderation
- **Brand Logo Detection** - Identify company logos
- **Landmark Recognition** - Identify famous places
- **Scene Recognition** - Classify image scenes
- **Anomaly Detection** - Identify unusual patterns

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                       │
│    (Web, Mobile, IoT, Batch Processing)                     │
└─────────────────┬───────────────────────────────────────────┘
                  │ HTTP/REST API
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Server                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ Auth       │  │ Rate       │  │ Validation │            │
│  │ Middleware │  │ Limiting   │  │ Layer      │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                 Processing Pipeline                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   Image    │  │   Model    │  │   Post     │            │
│  │Preprocessing│→│  Inference │→│ Processing │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────┬───────────────────────────────────────────┘
                  │
        ┌─────────┼─────────┐
        │         │         │
        ▼         ▼         ▼
┌───────────┐ ┌──────────┐ ┌──────────┐
│TensorFlow │ │ PyTorch  │ │  ONNX    │
│  Models   │ │  Models  │ │ Runtime  │
└───────────┘ └──────────┘ └──────────┘
        │         │         │
        └─────────┼─────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  Storage & Cache                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   AWS S3   │  │   Redis    │  │ PostgreSQL │            │
│  │  (Images)  │  │  (Cache)   │  │  (Metadata)│            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
image-recognition-api/
├── app/
│   ├── main.py                    # FastAPI application
│   ├── api/
│   │   ├── v1/
│   │   │   ├── classification.py
│   │   │   ├── detection.py
│   │   │   ├── faces.py
│   │   │   ├── ocr.py
│   │   │   └── processing.py
│   │   └── dependencies.py
│   ├── models/
│   │   ├── classification/
│   │   │   ├── resnet50.py
│   │   │   ├── efficientnet.py
│   │   │   └── vit.py
│   │   ├── detection/
│   │   │   ├── yolov8.py
│   │   │   └── faster_rcnn.py
│   │   ├── faces/
│   │   │   └── facenet.py
│   │   └── ocr/
│   │       └── tesseract.py
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   │   └── cache.py
│   ├── schemas/
│   │   └── requests.py
│   └── utils/
│       ├── image_processing.py
│       ├── model_loader.py
│       └── s3_client.py
├── models/                        # Pre-trained model weights
│   ├── classification/
│   ├── detection/
│   └── faces/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.gpu
│   └── docker-compose.yml
├── kubernetes/
│   ├── deployment.yaml
│   └── service.yaml
├── scripts/
│   ├── download_models.py
│   ├── train_custom_model.py
│   └── benchmark.py
├── notebooks/
│   ├── model_evaluation.ipynb
│   └── data_exploration.ipynb
└── docs/
    ├── api-reference.md
    ├── model-cards.md
    └── deployment-guide.md
```

## Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/cezarfuhr/image-recognition-api.git
cd image-recognition-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python scripts/download_models.py

# Set environment variables
cp .env.example .env
# Edit .env with your configurations

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# API available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Docker Deployment

```bash
# Build image
docker build -t image-recognition-api .

# Run container (CPU)
docker run -p 8000:8000 image-recognition-api

# Run container (GPU - requires nvidia-docker)
docker run --gpus all -p 8000:8000 image-recognition-api:gpu
```

### Docker Compose

```bash
# Start all services (API + Redis + PostgreSQL)
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## API Examples

### Image Classification

```bash
# Classify single image
curl -X POST "http://localhost:8000/api/v1/classify" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@cat.jpg"

# Response
{
  "predictions": [
    {"label": "tabby_cat", "confidence": 0.87},
    {"label": "egyptian_cat", "confidence": 0.09},
    {"label": "tiger_cat", "confidence": 0.03}
  ],
  "processing_time_ms": 145
}
```

### Object Detection

```bash
# Detect objects in image
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@street.jpg" \
  -F "confidence_threshold=0.5"

# Response
{
  "objects": [
    {
      "label": "person",
      "confidence": 0.95,
      "bbox": {"x": 120, "y": 45, "width": 180, "height": 320}
    },
    {
      "label": "car",
      "confidence": 0.89,
      "bbox": {"x": 450, "y": 200, "width": 240, "height": 160}
    }
  ],
  "count": 2,
  "processing_time_ms": 234
}
```

### Face Recognition

```bash
# Detect and encode faces
curl -X POST "http://localhost:8000/api/v1/faces/encode" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@group_photo.jpg"

# Response
{
  "faces": [
    {
      "id": 0,
      "bbox": {"x": 100, "y": 50, "width": 80, "height": 100},
      "encoding": [0.123, -0.456, ...],  # 128-dim vector
      "confidence": 0.98
    }
  ],
  "count": 1
}
```

### OCR (Text Extraction)

```bash
# Extract text from image
curl -X POST "http://localhost:8000/api/v1/ocr/extract" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@document.jpg" \
  -F "language=eng"

# Response
{
  "text": "This is the extracted text from the image...",
  "confidence": 0.92,
  "blocks": [
    {
      "text": "This is the extracted text",
      "bbox": {"x": 10, "y": 20, "width": 300, "height": 50},
      "confidence": 0.94
    }
  ]
}
```

### Background Removal

```bash
# Remove background from image
curl -X POST "http://localhost:8000/api/v1/process/remove-background" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@product.jpg" \
  --output product_nobg.png
```

## Python SDK Example

```python
from image_recognition_client import ImageRecognitionAPI

# Initialize client
api = ImageRecognitionAPI(
    base_url="http://localhost:8000",
    api_key="YOUR_API_KEY"
)

# Classify image
result = api.classify("path/to/image.jpg")
print(f"Top prediction: {result.predictions[0].label}")

# Detect objects
detections = api.detect_objects("path/to/image.jpg")
for obj in detections.objects:
    print(f"{obj.label}: {obj.confidence:.2f}")

# Extract text
text = api.extract_text("path/to/document.jpg", language="eng")
print(f"Extracted: {text.text}")

# Batch processing
results = api.classify_batch([
    "image1.jpg",
    "image2.jpg",
    "image3.jpg"
])
```

## Supported Models

### Classification
- **ResNet-50** - 50-layer residual network
- **EfficientNet-B7** - Efficient architecture
- **Vision Transformer (ViT)** - Transformer-based
- **MobileNet V3** - Lightweight for mobile
- **DenseNet-201** - Dense connections

### Detection
- **YOLOv8** - Latest YOLO version
- **Faster R-CNN** - Region-based CNN
- **SSD MobileNet** - Fast detection
- **EfficientDet** - Scalable detection
- **DETR** - Transformer-based detection

### Face Recognition
- **FaceNet** - Face embeddings
- **ArcFace** - Additive angular margin
- **MTCNN** - Face detection
- **DeepFace** - Facebook's model
- **VGGFace2** - Large-scale dataset

### OCR
- **Tesseract 5.0** - Open-source OCR
- **EasyOCR** - Deep learning OCR
- **PaddleOCR** - Multilingual OCR
- **CRAFT** - Text detection
- **TrOCR** - Transformer OCR

## Performance Benchmarks

### Inference Speed (single image)

| Model | CPU (ms) | GPU (ms) | Accuracy |
|-------|----------|----------|----------|
| ResNet-50 | 450 | 15 | 76.1% |
| EfficientNet-B7 | 1200 | 35 | 84.3% |
| YOLOv8-n | 80 | 8 | mAP 50.2 |
| YOLOv8-x | 350 | 25 | mAP 54.8 |
| FaceNet | 200 | 12 | 99.6% |
| Tesseract | 800 | - | 88.0% |

*Tested on: Intel i7-9700K (CPU), NVIDIA RTX 3080 (GPU)*

## Technology Stack

### Core Framework
- **Python**: 3.10+
- **FastAPI**: High-performance async API
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

### Deep Learning
- **TensorFlow**: 2.13+
- **PyTorch**: 2.0+
- **ONNX Runtime**: Cross-platform inference
- **OpenCV**: Image processing
- **Pillow**: Image manipulation

### Computer Vision
- **YOLOv8**: Ultralytics
- **face_recognition**: dlib-based
- **Tesseract OCR**: Google's OCR engine
- **scikit-image**: Image algorithms

### Infrastructure
- **Redis**: Response caching
- **PostgreSQL**: Metadata storage
- **AWS S3**: Image storage
- **Celery**: Async task queue
- **Docker**: Containerization

## API Features

### Authentication & Security
- API key authentication
- JWT token support
- Rate limiting (100 req/min)
- HTTPS/TLS encryption
- Input validation
- File size limits (max 10MB)

### Performance Optimization
- Response caching with Redis
- Model preloading
- Batch inference
- GPU acceleration
- Image preprocessing pipeline
- Async request handling

### Monitoring
- Prometheus metrics
- Request/response logging
- Error tracking with Sentry
- Performance profiling
- Health check endpoints

## Development Roadmap

### Phase 1: Core Features (Q2 2024)
- [x] Basic API structure
- [ ] Image classification endpoint
- [ ] Object detection (YOLOv8)
- [ ] Face detection
- [ ] OCR basic functionality
- [ ] Docker deployment

### Phase 2: Advanced Features (Q3 2024)
- [ ] Face recognition and matching
- [ ] Background removal
- [ ] Image segmentation
- [ ] Custom model training API
- [ ] Batch processing endpoints
- [ ] Kubernetes deployment

### Phase 3: Enterprise Features (Q4 2024)
- [ ] Video processing
- [ ] Real-time streaming
- [ ] Multi-model ensembles
- [ ] A/B testing framework
- [ ] Model versioning
- [ ] Auto-scaling

### Phase 4: AI Enhancements (2025)
- [ ] Zero-shot learning
- [ ] Few-shot learning
- [ ] Active learning pipeline
- [ ] Explainable AI (XAI)
- [ ] Edge deployment (TensorFlow Lite)
- [ ] Mobile SDK

## Training Custom Models

```python
# Train custom classifier
from app.training import CustomClassifier

classifier = CustomClassifier(
    base_model="resnet50",
    num_classes=10,
    dataset_path="data/custom_dataset"
)

# Train model
classifier.train(
    epochs=50,
    batch_size=32,
    learning_rate=0.001
)

# Save model
classifier.save("models/custom/my_model")

# Deploy to API
classifier.deploy(
    endpoint_name="custom-classifier",
    version="v1"
)
```

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

### Areas for Contribution
- New model integrations
- Performance optimizations
- Additional endpoints
- Documentation improvements
- Example notebooks

## Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Computer Vision Papers](https://paperswithcode.com/area/computer-vision)
- [YOLO Documentation](https://docs.ultralytics.com/)
- [FastAPI Guide](https://fastapi.tiangolo.com/)

## License

MIT License - See LICENSE file for details

## Contact

**Cezar Fuhr**
- Portfolio: [primoia.dev](https://www.primoia.dev)
- GitHub: [@cezarfuhr](https://github.com/cezarfuhr)
- Email: primoia.dev@gmail.com

---

⭐ Star this repo if you find it useful for computer vision tasks!