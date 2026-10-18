# LyricWala

A soft, interactive writing studio for your lyric diffusion model. Green `#1ED760`, ink `#121212`, white, and quiet neutral shades. Includes light/dark themes, an interactive vinyl record, streaming progress, cancellation, saved takes, copy, and text downloads.

The frontend and model service are separate. Your existing training repository is not changed.

## 1. Try the design immediately

Open `frontend/index.html` in a modern browser. No build step, packages, account, or network connection is required for this preview.

It starts in **demo mode** with original, illustrative sample lyrics. These are not outputs from your checkpoint. The short wait in this mode demonstrates the loading animation. Genre, mood and creativity are sent to the model in live mode; the offline samples are not a real generator.

## 2. Run the separate API

Python 3.10+ recommended. From the `backend` folder:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

In a second terminal, from the project root:

```bash
python -m http.server 5173 --bind 127.0.0.1 --directory frontend
```

In `frontend/index.html`, change the configuration near the start of the script:

```js
window.LYRICWALA_CONFIG ??= {
  mode: "api",
  apiBaseUrl: "http://127.0.0.1:8000"
};
```

Visit **http://127.0.0.1:5173**. The browser now calls FastAPI. The API itself defaults to demo mode, so you can test the whole connection before loading weights. API documentation: http://127.0.0.1:8000/docs.

Do not use a `file://` page for live API mode: serve it as above so the browser has a normal origin.

## 3. Connect your trained model

Use your working training environment, or install its exact PyTorch/Transformers versions in the API environment. `requirements-model.txt` provides broad starting requirements, but cannot guarantee compatibility with every checkpoint. Install the appropriate CUDA build of PyTorch for your machine.

Copy `backend/.env.example` to `backend/.env`. Set:

```dotenv
LYRICWALA_MODE=model
LYRICWALA_REPO=C:/projects/Diffusion_LyricGenerator
LYRICWALA_CHECKPOINT=C:/projects/Diffusion_LyricGenerator/checkpoints/best_model.pt
LYRICWALA_TOKENIZER=C:/projects/Diffusion_LyricGenerator/checkpoints
LYRICWALA_DEVICE=auto
LYRICWALA_STEPS=32
LYRICWALA_MIN_TIMESTEP=0.001
```

Use your actual paths. The adapter expects these existing repository files:

```
training_4/config.py
diffusion_model_3/model.py
```

It loads `BidirectionalDiffusionTransformer`, `model_state_dict`, and the saved tokenizer. The tokenizer folder must contain the tokenizer saved with your training run, including its added tokens. Vocabulary order matters, not just size.

For older checkpoints that did not save a tokenizer, `.env.example` explains an explicit legacy option. Only enable it if your model name and special-token order still match training exactly. If training sampled timesteps only from 0.1 to 1.0, set `LYRICWALA_MIN_TIMESTEP=0.1` for that checkpoint. Use the matching lower bound for newer training runs.

Restart the API. Model weights are loaded once, not on every click. A checkpoint/configuration error fails visibly at startup; the API never silently substitutes sample lyrics for a failed model.

### What the controls really do

- **Genre and mood** become part of the conditioning prompt. Whether the model follows them depends on its training.
- **Creative freedom** changes sampling temperature, from 0.4 to 1.2.
- **Length** sets 64, 128, or 176 output tokens, not a guaranteed number of lines. Overlong prompts receive a clear error rather than silent truncation.
- **Loading progress** comes from completed denoising steps. It measures steps, not exact seconds or lyric quality.
- **Cancel** stops between model passes; an in-progress GPU pass must finish first.

The adapter uses a fixed-length masked-token sampler with top-50 sampling and random-position unmasking. It does not add line breaks to training data, retrain your model, or fix poor checkpoint quality. The frontend displays the text your model returns. Tune the sampler in `backend/engine.py` if your training or inference scheme changes.

No real checkpoint is included. GPU generation and lyric quality must be verified with your own weights.

## API contract

`POST /api/generate`

```json
{"prompt":"A late-night drive towards a new beginning","genre":"Indie pop","mood":"Dreamy","length":"standard","creativity":50}
```

The response is a stream of newline-delimited JSON (`application/x-ndjson`), not one JSON document:

```json
{"type":"progress","progress":25,"stage":"Finding the feeling","step":8,"steps":32}
{"type":"result","title":"A dreamy little feeling","lyrics":"...","demo":false}
```

Failures before streaming use normal HTTP status codes: invalid input is 422; a busy model is 429. Failures after streaming starts use `{"type":"error","message":"..."}`. `GET /api/health` reports mode and busy status. Each request has a 180-second frontend timeout; increase it in `generate()` if your hardware needs longer.

## Files

```
frontend/index.html        UI, styles and browser logic; intentionally no build step
backend/app.py            FastAPI, streaming, cancellation and one-job concurrency guard
backend/schemas.py        Validated request contract
backend/engine.py         Separate demo engine and checkpoint adapter
backend/.env.example      Model paths and API configuration
backend/requirements.txt  Minimal API dependencies
backend/tests/            API tests; no model download required
tests/ui-smoke.cjs        Browser interaction checks
```

## Tests

From `backend`:

```bash
pip install pytest httpx
pytest -q
```

With the frontend served on port 5173, browser checks can be run with Node and Playwright installed:

```bash
npm install --no-save playwright
npx playwright install chromium
node tests/ui-smoke.cjs
```

Set `LYRICWALA_TEST_API=http://127.0.0.1:8000` to also exercise the live demo API. Run that service in demo mode for these tests. Browser screenshots are only written if `LYRICWALA_SCREENSHOTS` is set to an existing directory.

Delivery verification: 12 API tests passed, plus a real HTTP streaming/disconnect test that confirmed cancellation and the single-job guard. JavaScript syntax and HTML identifiers were checked. DOM interaction tests also passed for themes, the record, presets, controls, progress, cancellation, results, saving, search, reopening, and deletion, with no script errors. The included browser test suite could not run in the build environment because browser startup was blocked by its sandbox; visual and browser-layout verification remain to be run on your machine. Actual checkpoint/GPU inference was not tested because no checkpoint was supplied.

## Before public deployment

This is a working local application, not a hardened public GPU service. Use HTTPS, authentication, rate limits, request limits and a proper job queue before opening it to the internet. Configure explicit frontend origins in `.env`; CORS is not authentication. Use **one Uvicorn worker per model/GPU** with this implementation: its lock is process-local. A reverse proxy must allow streaming and disable response buffering for the generation route.

Saved lyrics stay in this browser's local storage; they are not a cloud account or backup. Download important takes. Prompts go only to the configured API in live mode. There are no analytics, external fonts, external image requests, or API secrets in the frontend.
