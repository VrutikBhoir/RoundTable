# Roundtable

Roundtable is a browser-based meeting experience for hosted, invitation-only
audio sessions. Hosts create a session and manage the lobby; participants join
from an invitation link, complete microphone setup and voice enrollment, and
connect to the live roundtable. The FastAPI backend provides realtime session
coordination and audio intelligence services such as voice activity detection,
transcription, diarization and speaker attribution.

## Project layout

```text
.
├── src/                    # React/Vite frontend
├── backend/                # FastAPI backend and audio-intelligence pipeline
├── public/                 # Static frontend assets
├── scripts/                # Development/test scripts
└── backend/requirements.txt
```

## Prerequisites

- Node.js with npm
- Python 3.10 or newer
- A working microphone for testing the participant flow
- A supported PyTorch runtime for the audio-intelligence features

The first backend startup may download model files used by Whisper, Silero VAD
and SpeechBrain. This can take longer than subsequent starts and requires
network access.

## Local development

### 1. Install frontend dependencies

From the repository root:

```bash
npm install
```

### 2. Create a Python environment and install backend dependencies

From the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install the backend packages:

```bash
python -m pip install -r backend/requirements.txt
```

### 3. Configure the backend

Copy [`backend/.env.example`](backend/.env.example) to `.env.local` in the
repository root and adjust the model settings if needed:

```dotenv
WHISPER_MODEL=small
WHISPER_LANGUAGE=en
WHISPER_DEVICE=auto
WHISPER_COMPUTE_TYPE=auto
```

Optional deployment-related settings include:

```dotenv
CORS_ORIGINS=http://localhost:5173
PUBLIC_APP_URL=http://localhost:5173
```

### 4. Start the backend

With the virtual environment active, run this from the repository root:

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

The API is available at <http://localhost:8000>. Health and interactive API
documentation are available at:

- <http://localhost:8000/health>
- <http://localhost:8000/docs>

### 5. Start the frontend

In a second terminal, from the repository root:

```bash
npm run dev
```

Open <http://localhost:5173>.

By default, the frontend uses same-origin HTTP requests for `/api` and connects
to the local backend WebSocket at `ws://localhost:8000`. For a remote backend,
create a root `.env.local` file with:

```dotenv
VITE_API_URL=https://your-api.example.com
VITE_WS_URL=wss://your-api.example.com
```

Do not commit `.env.local` or other files containing credentials.

## Available frontend scripts

```bash
npm run dev       # Start the Vite development server
npm run build     # Type-check and create a production build in dist/
npm run preview   # Preview the production build locally
```

## Backend tests

Run the backend tests from the repository root with the configured Python
environment:

```bash
python -m pytest backend
```

The tests cover room/session management, late joining, transcription behavior
and voice enrollment.

## Main application flows

- `/` — product landing page
- `/create-session` — create a host session
- `/host-lobby` — manage invitations, join requests and participants
- `/join` and `/join/:token` — preview and request access to an invitation
- `/microphone-setup` — microphone checks and voice enrollment
- `/roundtable` — live session experience
- `/results` — reserved placeholder for future meeting results

The backend exposes REST endpoints under `/api` for sessions, invitations,
participants and intelligence operations. Realtime session signaling is
provided by:

```text
/ws/sessions/{session_id}
```

Host-only operations use the `X-Host-Token` header. Participant operations use
the participant credentials returned after an invitation request is approved.

## Production build

Build the frontend with:

```bash
npm run build
```

Serve the generated `dist/` directory with a static host and run the FastAPI
application separately. Set `VITE_API_URL`, `VITE_WS_URL`, `CORS_ORIGINS` and
`PUBLIC_APP_URL` for the deployed origins, and ensure the deployment supports
WebSocket upgrades.
