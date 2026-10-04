export type EnrollmentProgress = "recording" | "uploading" | "verified" | "failed";

export type VoiceEnrollmentResult = {
  participantId: string;
  speechSeconds: number;
  recordingSeconds: number;
};

export async function enrollVoiceProfile(options: {
  stream: MediaStream;
  endpoint: string;
  credentials: Record<string, string>;
  seconds: number;
  onProgress?: (seconds: number) => void;
  onProcessing?: () => void;
}): Promise<VoiceEnrollmentResult> {
  const duration = options.seconds;
  if (typeof MediaRecorder === "undefined") throw new Error("Voice recording is unavailable in this browser.");
  const recorder = new MediaRecorder(options.stream);
  const chunks: Blob[] = [];
  const stopped = new Promise<void>((resolve, reject) => {
    recorder.ondataavailable = (event) => {
      if (event.data.size) chunks.push(event.data);
    };
    recorder.onerror = () => reject(new Error("Voice recording failed."));
    recorder.onstop = () => resolve();
  });
  recorder.start(250);
  const started = performance.now();
  await new Promise<void>((resolve) => {
    const timer = window.setInterval(() => {
      const elapsed = Math.min(duration, (performance.now() - started) / 1000);
      options.onProgress?.(elapsed);
      if (elapsed >= duration) {
        window.clearInterval(timer);
        recorder.stop();
        resolve();
      }
    }, 250);
  });
  await stopped;
  options.onProcessing?.();
  const body = new FormData();
  body.append("audio", new Blob(chunks, { type: recorder.mimeType || "audio/webm" }), "enrollment.webm");
  Object.entries(options.credentials).forEach(([key, value]) => body.append(key, value));
  const response = await fetch(options.endpoint, { method: "POST", body });
  if (!response.ok) {
    const detail = await response.json().catch(() => null) as { detail?: { message?: string } } | null;
    throw new Error(detail?.detail?.message ?? `Voice enrollment failed (${response.status}).`);
  }
  const result = await response.json() as {
    participant_id?: string;
    speech_seconds?: number;
    recording_seconds?: number;
  };
  if (
    typeof result.participant_id !== "string" ||
    typeof result.speech_seconds !== "number" ||
    typeof result.recording_seconds !== "number"
  ) {
    throw new Error("Voice enrollment returned an invalid response.");
  }
  return {
    participantId: result.participant_id,
    speechSeconds: result.speech_seconds,
    recordingSeconds: result.recording_seconds,
  };
}
