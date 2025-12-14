/**
 * Node.js version of the AssemblyAI streaming server:
 * - captures audio from a pre-configured Windows input device (via PortAudio/naudiodon)
 * - streams audio to AssemblyAI realtime API
 * - serves latest transcripts as JSON over a local HTTP server
 */
import { createServer } from "http";
import process from "process";
import { config as loadEnv } from "dotenv";
import portAudio from "naudiodon";
import { RealtimeClient } from "assemblyai";

loadEnv();

const API_KEY = process.env.ASSEMBLYAI_API_KEY;
const DEVICE_INDEX = Number(process.env.AUDIO_DEVICE_INDEX ?? 0);
const PORT = Number(process.env.TRANSCRIPT_SERVER_PORT ?? 8080);
const SAMPLE_RATE = Number(process.env.SAMPLE_RATE ?? 16000);

if (!API_KEY) {
  console.error("Missing ASSEMBLYAI_API_KEY in environment/.env");
  process.exit(1);
}

class TranscriptionState {
  constructor(maxHistory = 50) {
    this.maxHistory = maxHistory;
    this.status = "starting";
    this.sessionId = null;
    this.latest = null;
    this.history = [];
    this.partialCount = 0;
    this.finalCount = 0;
    this.error = null;
  }

  setStatus(status) {
    this.status = status;
  }

  setSession(sessionId) {
    this.sessionId = sessionId;
  }

  setError(message) {
    this.error = message;
    this.status = "error";
  }

  addTranscript(text, isFinal) {
    const entry = {
      text,
      type: isFinal ? "final" : "partial",
      timestamp: Date.now() / 1000,
    };
    this.latest = entry;
    if (isFinal) this.finalCount += 1;
    else this.partialCount += 1;
    this.history.push(entry);
    if (this.history.length > this.maxHistory) {
      this.history.shift();
    }
  }

  snapshot(extra = {}) {
    return {
      status: this.status,
      session_id: this.sessionId,
      latest: this.latest,
      partial_count: this.partialCount,
      final_count: this.finalCount,
      history: this.history,
      error: this.error,
      ...extra,
    };
  }
}

const state = new TranscriptionState();

// Configure AssemblyAI realtime client
const client = new RealtimeClient({
  token: API_KEY,
  sampleRate: SAMPLE_RATE,
});

client.on("open", ({ sessionId }) => {
  state.setSession(sessionId);
  state.setStatus("connected");
  console.log(`Connected to AssemblyAI (session ${sessionId})`);
});

client.on("transcript", (data) => {
  // Normalized flag for "final" vs "partial"
  const isFinal =
    data.message_type === "FinalTranscript" ||
    data.type === "final" ||
    data.final === true ||
    data.is_final === true;

  const text = data.text || data.transcript || "";
  if (!text) return;
  state.addTranscript(text, isFinal);
});

client.on("error", (err) => {
  const message = typeof err === "string" ? err : err?.message || "Unknown error";
  console.error("Streaming error:", message);
  state.setError(message);
});

client.on("close", () => {
  state.setStatus("stopped");
  console.log("AssemblyAI connection closed");
});

async function startStreaming() {
  console.log("Starting AssemblyAI realtime client...");
  await client.connect();
  state.setStatus("streaming");

  // Configure audio input using PortAudio
  const ai = new portAudio.AudioInput({
    deviceId: DEVICE_INDEX,
    channelCount: 1,
    sampleFormat: portAudio.SampleFormat16Bit,
    sampleRate: SAMPLE_RATE,
    highwaterMark: 4096,
  });

  ai.on("data", (chunk) => {
    // Forward audio to AssemblyAI
    client.sendAudio(chunk);
  });

  ai.on("error", (err) => {
    console.error("Audio input error:", err);
    state.setError(err?.message || String(err));
    shutdown();
  });

  ai.start();
  console.log(
    `Streaming audio from device ${DEVICE_INDEX} at ${SAMPLE_RATE} Hz; JSON at http://127.0.0.1:${PORT}`
  );

  return ai;
}

// Simple HTTP JSON endpoint
const server = createServer((_req, res) => {
  const payload = state.snapshot({ path: _req.url });
  const data = Buffer.from(JSON.stringify(payload));
  res.statusCode = 200;
  res.setHeader("Content-Type", "application/json");
  res.setHeader("Content-Length", data.length);
  res.end(data);
});

server.on("clientError", (err, socket) => {
  socket.end("HTTP/1.1 400 Bad Request\r\n\r\n");
  console.error("HTTP client error:", err);
});

let audioInput = null;

async function shutdown() {
  state.setStatus("stopping");
  try {
    server.close();
  } catch {
    /* ignore */
  }
  try {
    if (audioInput) {
      audioInput.quit();
    }
  } catch {
    /* ignore */
  }
  try {
    await client.close();
  } catch {
    /* ignore */
  }
}

process.on("SIGINT", () => {
  console.log("Received SIGINT, shutting down...");
  shutdown().finally(() => process.exit(0));
});
process.on("SIGTERM", () => {
  console.log("Received SIGTERM, shutting down...");
  shutdown().finally(() => process.exit(0));
});

(async () => {
  try {
    audioInput = await startStreaming();
    server.listen(PORT, "0.0.0.0");
  } catch (err) {
    console.error("Failed to start streaming:", err);
    state.setError(err?.message || String(err));
    process.exit(1);
  }
})();
