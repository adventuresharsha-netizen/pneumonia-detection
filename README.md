# Pneumonia Detection App

Streamlit app that loads the best model from the final project notebook, accepts a chest X-ray (DICOM, PNG or JPEG) and shows the predicted class and pneumonia probability.

> Academic decision-support prototype. Not a clinical diagnostic tool.

## Folder layout

```text
app.py
requirements.txt
Dockerfile
.dockerignore
.devcontainer/devcontainer.json
model/best_model.keras
model/model_config.json
```

## Dependencies

- Backend: tensorflow-cpu, pydicom, opencv-python-headless, numpy
- Frontend: streamlit, pillow

## Run in GitHub Codespaces

```bash
docker build -t pneumonia-app .
docker run -p 8501:8501 pneumonia-app
```

Open the Ports tab, copy the forwarded URL for port 8501, and upload a test image.

## Run locally without Docker

```bash
pip install -r requirements.txt
streamlit run app.py
```
