"""
Lightweight streaming server that reuses the AssemblyAI/vMix helpers to:
- connect to a pre-configured Windows audio input (via .env)
- stream audio to AssemblyAI's Universal Streaming API
- expose the latest transcripts as JSON over a local HTTP server
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
)

# Make local helpers importable (support folder inside here or sibling in parent)
BASE_DIR = Path(__file__).resolve().parent
helper_paths = [
    BASE_DIR / "assembly-vmix",
    BASE_DIR.parent / "assembly-vmix",
]
for candidate in helper_paths:
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

from config import Config, ConfigurationError  # noqa: E402
from audio_stream import MicrophoneStream, AudioStreamError  # noqa: E402


def setup_logging() -> None:
    """Configure simple console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


class TranscriptionState:
    """Thread-safe store for the latest transcript data."""

    def __init__(self, max_history: int = 50):
        self._lock = threading.Lock()
        self.latest: Optional[Dict[str, Any]] = None
        self.history: list[Dict[str, Any]] = []
        self.status: str = "starting"
        self.session_id: Optional[str] = None
        self.partial_count = 0
        self.final_count = 0
        self.error: Optional[str] = None
        self._max_history = max_history

    def set_status(self, status: str) -> None:
        with self._lock:
            self.status = status

    def set_session(self, session_id: str) -> None:
        with self._lock:
            self.session_id = session_id

    def set_error(self, message: str) -> None:
        with self._lock:
            self.error = message
            self.status = "error"

    def add_transcript(self, text: str, is_final: bool) -> None:
        """Store the latest transcript and keep a short history."""
        entry = {
            "text": text,
            "type": "final" if is_final else "partial",
            "timestamp": time.time(),
        }
        with self._lock:
            self.latest = entry
            if is_final:
                self.final_count += 1
            else:
                self.partial_count += 1
            self.history.append(entry)
            if len(self.history) > self._max_history:
                self.history.pop(0)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "session_id": self.session_id,
                "latest": self.latest,
                "partial_count": self.partial_count,
                "final_count": self.final_count,
                "history": list(self.history),
                "error": self.error,
            }


def make_http_handler(state: TranscriptionState):
    """Create an HTTP handler class bound to the shared state."""

    class Handler(BaseHTTPRequestHandler):
        def _write_json(self, payload: Dict[str, Any], code: int = 200) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            payload = state.snapshot()
            payload["path"] = self.path
            self._write_json(payload)

        def log_message(self, fmt: str, *args: Any) -> None:
            logging.info("HTTP %s - %s", self.client_address[0], fmt % args)

    return Handler


def create_streaming_client(state: TranscriptionState) -> StreamingClient:
    """Configure the AssemblyAI streaming client and event handlers."""
    client = StreamingClient(
        StreamingClientOptions(
            api_key=Config.ASSEMBLYAI_API_KEY,
            api_host="streaming.assemblyai.com",
        )
    )

    def on_begin(_client, event):
        state.set_session(event.id)
        state.set_status("connected")
        logging.info("Connected to AssemblyAI, session id=%s", event.id)

    def on_turn(_client, event):
        if not event.transcript:
            return
        state.add_transcript(event.transcript, event.end_of_turn)

    def on_error(_client, error):
        message = str(error)
        state.set_error(message)
        logging.error("Streaming error: %s", message)

    def on_terminated(_client, event):
        logging.info(
            "Streaming terminated after %.2f seconds of audio", event.audio_duration_seconds
        )
        state.set_status("stopped")

    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Error, on_error)
    client.on(StreamingEvents.Termination, on_terminated)
    return client


def stream_audio_loop(state: TranscriptionState, stop_event: threading.Event) -> None:
    """Read audio from the configured device and push to AssemblyAI until stopped."""
    client: Optional[StreamingClient] = None
    try:
        client = create_streaming_client(state)
        client.connect(
            StreamingParameters(
                sample_rate=Config.SAMPLE_RATE,
                format_turns=True,
            )
        )
        state.set_status("streaming")
        logging.info("Streaming audio from device index %s", Config.get_device_index())
        with MicrophoneStream() as mic:
            while not stop_event.is_set():
                audio_chunk = mic.read()
                client.stream(audio_chunk)
        logging.info("Stopping audio stream...")
    except AudioStreamError as e:
        state.set_error(str(e))
        logging.error("Audio error: %s", e)
    except Exception as e:  # pragma: no cover - defensive logging
        state.set_error(str(e))
        logging.exception("Unexpected error in streaming loop: %s", e)
    finally:
        if client:
            try:
                client.disconnect(terminate=True)
            except Exception:
                logging.debug("Streaming client disconnect encountered an error", exc_info=True)


def main() -> int:
    load_dotenv()
    setup_logging()

    try:
        Config.validate()
    except ConfigurationError as e:
        logging.error("Configuration error: %s", e)
        return 1

    state = TranscriptionState()
    stop_event = threading.Event()

    stream_thread = threading.Thread(
        target=stream_audio_loop, args=(state, stop_event), daemon=True
    )
    stream_thread.start()

    port = int(os.getenv("TRANSCRIPT_SERVER_PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), make_http_handler(state))
    logging.info("Serving transcript JSON at http://127.0.0.1:%s", port)

    def handle_shutdown(signum, _frame):
        logging.info("Received signal %s, shutting down...", signum)
        stop_event.set()
        server.shutdown()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        server.serve_forever()
    finally:
        stop_event.set()
        stream_thread.join(timeout=5)
        logging.info("Shutdown complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
