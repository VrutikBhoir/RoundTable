/* Live transcription feed: local mic -> rolling PCM windows -> backend pipeline.
 * Independent of WebRTC transport (which keeps carrying live audio);
 * results come back over the existing signaling socket as transcript_event.
 *
 * Windows are 1.5 s and non-overlapping: the backend VAD splits them into
 * speech regions and accumulates enough speech for one Whisper call. This
 * keeps live captions responsive while still filtering silence and scraps.
 */

const WORKLET_CODE = `
class FeedCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.carry = 0;
    this.buf = [];
  }
  process(inputs) {
    const ch = inputs && inputs[0] && inputs[0][0];
    if (ch) {
      for (let i = 0; i < ch.length; i++) {
        this.carry += ch[i];
        this._n = (this._n || 0) + 1;
        if (this._n >= this.ratio) {
          this.buf.push(this.carry / this._n);
          this.carry = 0;
          this._n = 0;
          if (this.buf.length >= 1600) {
            this.port.postMessage(new Float32Array(this.buf.splice(0, 1600)));
          }
        }
      }
    }
    return true;
  }
}
registerProcessor("rt-feed-capture", FeedCapture);
`;

// A shorter cadence reduces capture-to-caption latency. The backend still
// buffers speech regions, so Whisper is not called for every AudioWorklet
// chunk or for digital silence.
const WINDOW_S = 1.5;

export type FeedCredentials =
  | { participant_id: string; participant_token: string }
  | { host_token: string };

async function blobToB64(buf: Float32Array): Promise<string> {
  const bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  let binary = "";
  const CHUNK = 8192;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

export function startTranscriptionFeed(options: {
  stream: MediaStream;
  sessionId: string;
  credentials: FeedCredentials;
  apiBase: string;
  onError?: (message: string) => void;
}): () => void {
  const { stream, sessionId, credentials, apiBase } = options;
  const context = new AudioContext();
  let node: AudioWorkletNode | null = null;
  let stopped = false;
  let posting: Promise<void> | null = null;
  const feedT0 = Date.now();
  let acc = new Float32Array(0);
  let windowStartS = 0;
  let workletChunks = 0;
  let nonZeroWorkletChunks = 0;
  let uploadedWindows = 0;
  const participantId = "participant_id" in credentials ? credentials.participant_id : "host";

  const debug = (message: string, details: Record<string, unknown>) => {
    if (import.meta.env.DEV) console.debug(message, details);
  };

  const track = stream.getAudioTracks()[0];
  debug("[MICROPHONE]", {
    participant_id: participantId,
    stream_active: stream.active,
    audio_track_count: stream.getAudioTracks().length,
    track_ready_state: track?.readyState ?? "missing",
    track_enabled: track?.enabled ?? false,
    track_muted: track?.muted ?? false,
    audio_context_state: context.state,
  });

  const postWindow = (pcm: Float32Array, t0: number) => {
    const payload = { ...credentials, t0, pcm_b64: "" };
    const run = async () => {
      try {
        (payload as Record<string, unknown>).pcm_b64 = await blobToB64(pcm);
        const nonZero = pcm.reduce((count, value) => count + (value !== 0 ? 1 : 0), 0);
        const peak = pcm.reduce((current, value) => Math.max(current, Math.abs(value)), 0);
        const rms = Math.sqrt(pcm.reduce((sum, value) => sum + value * value, 0) / Math.max(1, pcm.length));
        uploadedWindows += 1;
        debug("[AUDIO UPLOAD]", {
          participant_id: participantId,
          segment_duration_s: pcm.length / 16000,
          pcm_sample_count: pcm.length,
          sample_rate: 16000,
          rms,
          peak_amplitude: peak,
          non_zero_percent: (nonZero / Math.max(1, pcm.length)) * 100,
          base64_length: String((payload as { pcm_b64: string }).pcm_b64).length,
          upload_index: uploadedWindows,
        });
        const res = await fetch(
          `${apiBase}/api/sessions/${encodeURIComponent(sessionId)}/audio-segments`,
          { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }
        );
        debug("[AUDIO UPLOAD RESULT]", {
          participant_id: participantId,
          status: res.status,
          accepted: res.ok,
        });
        if (!res.ok && options.onError) {
          options.onError(`transcription upload failed (${res.status})`);
        }
      } catch {
        // Transient network failure: skip this window, keep streaming.
      }
    };
    posting = run().finally(() => {
      if (posting !== null) posting = null;
    });
  };

  const onChunk = (chunk: Float32Array) => {
    if (stopped) return;
    workletChunks += 1;
    const chunkNonZero = chunk.some((value) => value !== 0);
    if (chunkNonZero) nonZeroWorkletChunks += 1;
    if (workletChunks === 1 || workletChunks % 25 === 0) {
      debug("[AUDIOWORKLET PCM]", {
        participant_id: participantId,
        chunk_sample_count: chunk.length,
        sample_rate: 16000,
        non_zero: chunkNonZero,
        chunks_received: workletChunks,
        non_zero_chunks: nonZeroWorkletChunks,
        audio_context_state: context.state,
      });
    }
    const merged = new Float32Array(acc.length + chunk.length);
    merged.set(acc);
    merged.set(chunk, acc.length);
    acc = merged;
    const need = WINDOW_S * 16000;
    while (acc.length >= need) {
      const window = acc.slice(0, need);
      acc = acc.slice(need);
      const t0 = windowStartS;
      windowStartS += WINDOW_S;
      // Skip digital silence locally so the backend never sees it.
      let peak = 0;
      for (let i = 0; i < window.length; i += 7) {
        const v = Math.abs(window[i]);
        if (v > peak) peak = v;
      }
      if (peak > 0.008) postWindow(window, t0);
      else {
        debug("[AUDIO WINDOW SKIPPED]", {
          participant_id: participantId,
          segment_duration_s: window.length / 16000,
          pcm_sample_count: window.length,
          sample_rate: 16000,
          peak_amplitude: peak,
          reason: "local_peak_gate",
        });
      }
    }
  };

  (async () => {
    try {
      const url = URL.createObjectURL(new Blob([WORKLET_CODE], { type: "application/javascript" }));
      try {
        await context.audioWorklet.addModule(url);
      } finally {
        URL.revokeObjectURL(url);
      }
      if (stopped) {
        void context.close().catch(() => undefined);
        return;
      }
      const source = context.createMediaStreamSource(stream);
      node = new AudioWorkletNode(context, "rt-feed-capture", { numberOfOutputs: 0 });
      node.port.onmessage = (event: MessageEvent) => {
        const data = event.data as Float32Array;
        if (data && data.length) onChunk(new Float32Array(data));
      };
      source.connect(node);
      debug("[AUDIOWORKLET READY]", {
        participant_id: participantId,
        audio_context_state: context.state,
        track_ready_state: track?.readyState ?? "missing",
        track_enabled: track?.enabled ?? false,
        feed_start_ms: feedT0,
      });
    } catch {
      if (options.onError) options.onError("local transcription capture unavailable in this browser");
    }
  })();

  return () => {
    stopped = true;
    try {
      node?.disconnect();
    } catch {
      // Already torn down.
    }
    node = null;
    void context.close().catch(() => undefined);
  };
}
