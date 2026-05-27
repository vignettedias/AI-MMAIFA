# Image Authenticity Frontend

React + Vite + Tailwind CSS frontend for the multi-modal AI image authenticity backend.

## Run Locally

Start the Python backend from the repo root:

```powershell
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Start the frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The Vite dev server proxies frontend API calls from `/api/predict` to `http://127.0.0.1:8000/predict`.

## Optional API Override

Use a custom backend URL:

```powershell
$env:VITE_API_BASE_URL="http://127.0.0.1:8000"
npm run dev
```

## Validate

```powershell
npm run lint
npm run build
```
