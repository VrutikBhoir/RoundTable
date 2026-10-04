/* Small, honest Web Audio helpers for the Microphone Setup page.
 * The test path only analyses live audio locally (AnalyserNode).
 * Nothing is recorded, stored, or sent to the backend. */

export type MicrophoneFailure = "denied" | "no-device" | "unsupported" | "unavailable";

export function microphoneSupported(): boolean {
  return (
    typeof navigator !== "undefined" &&
    !!navigator.mediaDevices &&
    typeof navigator.mediaDevices.getUserMedia === "function"
  );
}

export function classifyMicError(error: unknown): MicrophoneFailure {
  if (!microphoneSupported()) return "unsupported";
  const name = error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") return "denied";
  if (name === "NotFoundError" || name === "OverconstrainedError") return "no-device";
  if (name === "NotReadableError" || name === "AbortError") return "unavailable";
  return "unavailable";
}

export async function requestMicrophone(deviceId?: string): Promise<MediaStream> {
  if (!microphoneSupported()) {
    throw new DOMException("Microphone not supported", "NotSupportedError");
  }
  const audio: MediaTrackConstraints = {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
    ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
  };
  return navigator.mediaDevices.getUserMedia({
    audio,
  });
}

export function microphoneProcessingStatus(stream: MediaStream): "ON" | "OFF" | "UNAVAILABLE" {
  const track = stream.getAudioTracks()[0];
  if (!track) return "UNAVAILABLE";
  const settings = track.getSettings();
  if (
    settings.noiseSuppression === true &&
    settings.echoCancellation === true &&
    settings.autoGainControl === true
  ) return "ON";
  if (
    settings.noiseSuppression === false ||
    settings.echoCancellation === false ||
    settings.autoGainControl === false
  ) return "OFF";
  return "UNAVAILABLE";
}

export async function listAudioInputs(): Promise<MediaDeviceInfo[]> {
  if (!microphoneSupported()) return [];
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((d) => d.kind === "audioinput");
}

/** Friendly display name. Never exposes device IDs. */
export function friendlyMicLabel(device: MediaDeviceInfo, index: number, total: number): string {
  if (device.label) return device.label;
  if (total === 1) return "Phone Microphone";
  return index === 0 ? "Default microphone" : `Microphone ${index + 1}`;
}

export function stopStream(stream: MediaStream | null) {
  stream?.getTracks().forEach((track) => {
    try {
      track.stop();
    } catch {
      // Already stopped.
    }
  });
}

export type LevelMonitor = {
  /** Normalized 0..1 audio energy. Cheap to call every animation frame. */
  getLevel: () => number;
  dispose: () => void;
};

/** MediaStream -> AudioContext -> AnalyserNode. Local analysis only. */
export function createLevelMonitor(stream: MediaStream): LevelMonitor {
  const Context = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Context) throw new Error("AudioContext unavailable");
  const context = new Context();
  const source = context.createMediaStreamSource(stream);
  const analyser = context.createAnalyser();
  analyser.fftSize = 2048;
  source.connect(analyser);
  const data = new Uint8Array(analyser.fftSize);

  return {
    getLevel: () => {
      analyser.getByteTimeDomainData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i += 2) {
        const v = (data[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / (data.length / 2));
      return Math.min(1, Math.max(0, rms * 3));
    },
    dispose: () => {
      try {
        source.disconnect();
      } catch {
        // Already disconnected.
      }
      void context.close().catch(() => undefined);
    },
  };
}
