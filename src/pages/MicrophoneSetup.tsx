import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, CheckCircle2, Mic, MonitorSmartphone } from "lucide-react";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import SetupProgress from "../components/mic/SetupProgress";
import ConnectionStatus from "../components/mic/ConnectionStatus";
import MicrophonePermission from "../components/mic/MicrophonePermission";
import MicrophoneSelector from "../components/mic/MicrophoneSelector";
import MicrophoneTest, { type MicTestState } from "../components/mic/MicrophoneTest";
import AudioLevelMeter from "../components/mic/AudioLevelMeter";
import SeatPositionSelector, { type SeatId } from "../components/mic/SeatPositionSelector";
import {
  classifyMicError,
  createLevelMonitor,
  listAudioInputs,
  microphoneSupported,
  requestMicrophone,
  stopStream,
  type LevelMonitor,
  type MicrophoneFailure,
} from "../lib/audio";
import { apiBaseUrl, getVoiceStatus } from "../lib/api";
import { enrollVoiceProfile } from "../lib/voiceEnrollment";

type ParticipantSession = {
  invitationToken?: string;
  requestId?: string;
  sessionId?: string | null;
  participantId?: string | null;
  participantToken?: string | null;
  displayName?: string;
  sessionName?: string | null;
};

type PermissionState = "unknown" | "requesting" | "granted" | MicrophoneFailure;
type EnrollmentState = "idle" | "recording" | "processing" | "ready" | "failed" | "skipped";

const TEST_DURATION_MS = 4000;
const PASS_THRESHOLD = 0.04;

function formatDuration(seconds: number): string {
  const wholeSeconds = Math.max(0, Math.floor(seconds));
  return `00:${String(wholeSeconds).padStart(2, "0")}`;
}

function formatEnrollmentError(message: string): string {
  if (message.startsWith("Not enough speech detected.")) {
    return message;
  }
  if (/insufficient_speech/i.test(message)) {
    return "Not enough speech detected.";
  }
  return message;
}

function readParticipantSession(): ParticipantSession | null {
  try {
    const raw = sessionStorage.getItem("roundtable.participantSession");
    return raw ? (JSON.parse(raw) as ParticipantSession) : null;
  } catch {
    return null;
  }
}

export default function MicrophoneSetup() {
  const navigate = useNavigate();
  const [participant] = useState<ParticipantSession | null>(() => readParticipantSession());
  const [permission, setPermission] = useState<PermissionState>(() =>
    microphoneSupported() ? "unknown" : "unsupported"
  );
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string | null>(null);
  const [test, setTest] = useState<MicTestState>("idle");
  const [seat, setSeat] = useState<SeatId | null>(() => {
    try {
      return (sessionStorage.getItem("roundtable.seatPosition") as SeatId) || null;
    } catch {
      return null;
    }
  });
  const [seatSkipped, setSeatSkipped] = useState(false);
  const [disconnected, setDisconnected] = useState(false);
  const [enrollment, setEnrollment] = useState<EnrollmentState>("idle");
  const [enrollmentProgress, setEnrollmentProgress] = useState(0);
  const [speechSeconds, setSpeechSeconds] = useState<number | null>(null);
  const [enrollmentError, setEnrollmentError] = useState<string | null>(null);
  const [enrollmentDuration, setEnrollmentDuration] = useState<number | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const monitorRef = useRef<LevelMonitor | null>(null);
  const testRaf = useRef(0);

  const getLevel = useCallback(() => monitorRef.current?.getLevel() ?? 0, []);

  /* Stop the local stream when the page is abandoned. */
  useEffect(() => {
    if (!participant?.sessionId || !participant.participantId || !participant.participantToken) return;
    let active = true;
    void getVoiceStatus(participant.sessionId, {
      participant_id: participant.participantId,
      participant_token: participant.participantToken,
    }).then((status) => {
      console.debug("[voice] setup profile lookup", {
        session_id: participant.sessionId,
        participant_id: participant.participantId,
        status: status.status,
      });
      if (!active) return;
      setEnrollmentDuration(status.enrollment_duration_seconds);
      if (status.status === "ready") setEnrollment("ready");
    }).catch((error) => {
      console.warn("[voice] setup profile lookup failed", {
        session_id: participant.sessionId,
        participant_id: participant.participantId,
        error: error instanceof Error ? error.message : error,
      });
    });
    return () => {
      active = false;
    };
  }, [participant?.sessionId, participant?.participantId, participant?.participantToken]);

  useEffect(() => {
    return () => {
      cancelAnimationFrame(testRaf.current);
      monitorRef.current?.dispose();
      monitorRef.current = null;
      stopStream(streamRef.current);
      streamRef.current = null;
    };
  }, []);

  const attachStream = useCallback(async (stream: MediaStream) => {
    monitorRef.current?.dispose();
    stopStream(streamRef.current);
    streamRef.current = stream;
    setDisconnected(false);
    try {
      monitorRef.current = createLevelMonitor(stream);
    } catch {
      monitorRef.current = null;
    }
    try {
      setDevices(await listAudioInputs());
    } catch {
      // Device list is a nicety — the stream itself is what matters.
    }
    stream.getTracks().forEach((track) => {
      track.onended = () => {
        if (streamRef.current === stream) setDisconnected(true);
      };
    });
  }, []);

  async function enableMicrophone(selectedDeviceId?: string | null) {
    setPermission("requesting");
    setTest((t) => (t === "passed" || t === "quiet" ? "idle" : t));
    try {
      const stream = await requestMicrophone(selectedDeviceId ?? undefined);
      await attachStream(stream);
      const inputs = await listAudioInputs().catch(() => [] as MediaDeviceInfo[]);
      const activeId = stream.getAudioTracks()[0]?.getSettings().deviceId ?? inputs[0]?.deviceId ?? null;
      setDeviceId(activeId);
      setPermission("granted");
    } catch (error) {
      stopStream(streamRef.current);
      streamRef.current = null;
      setPermission(classifyMicError(error));
    }
  }

  async function switchDevice(nextId: string) {
    setDeviceId(nextId);
    setTest("idle");
    try {
      const stream = await requestMicrophone(nextId);
      await attachStream(stream);
    } catch (error) {
      setPermission(classifyMicError(error));
    }
  }

  function startTest() {
    if (!monitorRef.current) return;
    setTest("testing");
    const samples: number[] = [];
    const started = performance.now();
    const collect = () => {
      samples.push(monitorRef.current?.getLevel() ?? 0);
      if (performance.now() - started >= TEST_DURATION_MS) {
        const peak = samples.reduce((max, v) => Math.max(max, v), 0);
        setTest(peak >= PASS_THRESHOLD ? "passed" : "quiet");
        return;
      }
      testRaf.current = requestAnimationFrame(collect);
    };
    testRaf.current = requestAnimationFrame(collect);
  }

  function chooseSeat(id: SeatId) {
    setSeat(id);
    try {
      sessionStorage.setItem("roundtable.seatPosition", id);
    } catch {
      // Optional preference — safe to ignore.
    }
  }

  async function startVoiceEnrollment() {
    if (!streamRef.current || !participant?.sessionId || !enrollmentDuration) return;
    if (!participant.participantId || !participant.participantToken) {
      setEnrollmentError("Your participant identity is missing. Return to the join page and request access again.");
      setEnrollment("failed");
      return;
    }
    setEnrollment("recording");
    setEnrollmentProgress(0);
    setSpeechSeconds(null);
    setEnrollmentError(null);
    try {
      const result = await enrollVoiceProfile({
        stream: streamRef.current,
        endpoint: `${apiBaseUrl()}/api/sessions/${encodeURIComponent(participant.sessionId)}/voice-enrollment`,
        credentials: {
          participant_id: participant.participantId,
          participant_token: participant.participantToken,
        },
        seconds: enrollmentDuration,
        onProgress: setEnrollmentProgress,
        onProcessing: () => setEnrollment("processing"),
      });
      console.debug("[voice] setup enrollment response", {
        session_id: participant.sessionId,
        participant_id: result.participantId,
        speech_seconds: result.speechSeconds,
        recording_seconds: result.recordingSeconds,
      });
      setSpeechSeconds(result.speechSeconds);
      setEnrollment("ready");
    } catch (error) {
      setEnrollmentError(error instanceof Error ? error.message : "Voice enrollment failed.");
      setEnrollment("failed");
    }
  }

  function skipVoiceEnrollment() {
    setEnrollment("skipped");
    setEnrollmentError(null);
  }

  // 01 microphone → 02 test → 03 voice profile → 04 position → 05 ready
  const voiceComplete = enrollment === "ready" || enrollment === "skipped";
  const stepIndex = permission !== "granted" ? 0 : test !== "passed" ? 1 : !voiceComplete ? 2 : !(seat || seatSkipped) ? 3 : 4;
  const ready = permission === "granted" && test === "passed" && voiceComplete && !disconnected;

  function continueToLiveSession() {
    if (!participant || !participant.sessionId) {
      navigate("/join");
      return;
    }
    const sessionState = {
      sessionId: participant.sessionId,
      participantId: participant.participantId ?? null,
      participantToken: participant.participantToken ?? null,
      displayName: participant.displayName ?? "Participant",
      sessionName: participant.sessionName ?? "Roundtable Session",
      role: "participant",
    };
    navigate("/roundtable", { state: sessionState, replace: true });
  }

  return (
    <div className="min-h-screen bg-[#F7F7F5] text-[#111111]">
      <Navbar />
      <main className="mx-auto w-full max-w-[600px] px-5 pb-24 pt-[120px] md:pt-[150px]">
        <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
          <div className="text-center">
            <p className="inline-flex items-center gap-2 rounded-full border border-[#E4E4E0] bg-white px-4 py-1.5 font-mono text-[11px] uppercase tracking-[0.16em] text-[#666]">
              <Mic className="h-3.5 w-3.5 text-[#635BFF]" /> Microphone setup
            </p>
            <h1 className="editorial-tight mt-5 text-[36px] font-[700] sm:text-[48px]">Let&apos;s connect your microphone.</h1>
            <p className="mx-auto mt-3 max-w-[460px] text-[15px] leading-relaxed text-[#666]">
              Your phone will act as one of the microphones helping Roundtable understand the conversation.
            </p>
            {participant?.sessionName && (
              <div className="mt-4">
                <ConnectionStatus sessionName={participant.sessionName} />
              </div>
            )}
          </div>

          <div className="mt-8 rounded-[24px] border border-[#E4E4E0] bg-white p-5 sm:p-7">
            <SetupProgress current={stepIndex} />

            <div className="mt-6" aria-live="polite">
              <AnimatePresence mode="wait">
                {!participant ? (
                  <motion.div
                    key="no-session"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    transition={{ duration: 0.3 }}
                    className="rounded-2xl border border-[#E4E4E0] p-6 text-center"
                  >
                    <h2 className="text-[18px] font-bold">Join a session first</h2>
                    <p className="mx-auto mt-2 max-w-[380px] text-[14px] leading-relaxed text-[#666]">
                      Microphone setup is for approved participants. Scan your host&apos;s QR code to join, then come back here.
                    </p>
                    <Link
                      to="/join"
                      className="mt-5 inline-flex min-h-[48px] items-center justify-center rounded-full bg-[#111111] px-7 py-3 text-[14px] font-semibold text-white transition hover:-translate-y-[1px]"
                    >
                      Go to Join Session
                    </Link>
                  </motion.div>
                ) : permission === "unsupported" ? (
                  <motion.div
                    key="unsupported"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    transition={{ duration: 0.3 }}
                    className="rounded-2xl border border-[#E4E4E0] p-6 text-center"
                  >
                    <MonitorSmartphone className="mx-auto h-8 w-8 text-[#8A8A86]" strokeWidth={1.6} />
                    <h2 className="mt-3 text-[18px] font-bold">This browser cannot access your microphone.</h2>
                    <p className="mx-auto mt-2 max-w-[380px] text-[14px] leading-relaxed text-[#666]">
                      Try a recent version of Chrome, Edge, Firefox, or Safari on your phone.
                    </p>
                  </motion.div>
                ) : permission !== "granted" ? (
                  <motion.div
                    key="permission"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    transition={{ duration: 0.3 }}
                  >
                    <MicrophonePermission
                      state={permission === "requesting" ? "requesting" : permission === "unknown" ? "idle" : permission}
                      onAllow={() => void enableMicrophone(deviceId)}
                    />
                  </motion.div>
                ) : (
                  <motion.div
                    key="setup"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    transition={{ duration: 0.3 }}
                    className="space-y-5"
                  >
                    <div className="flex items-center gap-2 rounded-2xl bg-[#18A874]/10 px-4 py-3" role="status">
                      <CheckCircle2 className="h-5 w-5 shrink-0 text-[#18A874]" />
                      <p className="text-[14px] font-semibold">Microphone connected</p>
                    </div>

                    {disconnected && (
                      <div className="rounded-2xl border border-[#E8D9A8] bg-[#FDF6E3] p-4" role="alert">
                        <p className="text-[14px] font-semibold">Microphone disconnected.</p>
                        <button
                          type="button"
                          onClick={() => void enableMicrophone(deviceId)}
                          className="mt-3 inline-flex min-h-[44px] items-center justify-center rounded-full bg-[#111111] px-5 py-2.5 text-[13px] font-semibold text-white transition hover:-translate-y-[1px]"
                        >
                          Reconnect
                        </button>
                      </div>
                    )}

                    <section aria-label="Choose your microphone">
                      <h2 className="text-[16px] font-bold tracking-[-0.01em]">Choose your microphone</h2>
                      <div className="mt-3">
                        <MicrophoneSelector devices={devices} selectedId={deviceId} onSelect={(id) => void switchDevice(id)} />
                      </div>
                      <div className="mt-4">
                        <AudioLevelMeter getLevel={getLevel} live />
                      </div>
                    </section>

                    <MicrophoneTest
                      state={test}
                      getLevel={getLevel}
                      onStart={startTest}
                      onRetry={() => {
                        setTest("idle");
                        startTest();
                      }}
                    />

                    {test === "passed" && !voiceComplete && (
                      <motion.section
                        aria-label="Voice profile"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3 }}
                        className="rounded-2xl border border-[#E4E4E0] p-5"
                      >
                        <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#8A8A86]">Voice profile</p>
                        <h2 className="mt-2 text-[20px] font-bold">Create your voice profile</h2>
                        <p className="mt-2 text-[14px] leading-relaxed text-[#666]">
                          Speak naturally for a few seconds. Roundtable uses this profile as one signal when identifying speakers.
                        </p>

                        {enrollment === "recording" && (
                          <div className="mt-5">
                            <div className="flex items-center justify-between text-[14px] font-semibold">
                              <span>Recording voice profile</span>
                              <span className="font-mono tabular-nums">{formatDuration(enrollmentProgress)} / {formatDuration(enrollmentDuration ?? 0)}</span>
                            </div>
                            <div className="mt-4">
                              <AudioLevelMeter getLevel={getLevel} live label="Voice profile microphone level" />
                            </div>
                          </div>
                        )}

                        {enrollment === "processing" && (
                          <p className="mt-5 rounded-xl bg-[#F7F7F5] px-4 py-3 text-[14px] font-semibold" role="status">
                            Processing voice profile...
                          </p>
                        )}

                        {enrollment === "failed" && enrollmentError && (
                          <div className="mt-5 rounded-xl border border-[#E8B4B4] bg-[#FFF5F5] p-4 text-[14px]" role="alert">
                            <p className="font-semibold">{formatEnrollmentError(enrollmentError)}</p>
                            {enrollmentError.startsWith("Not enough speech detected.") && (
                              <p className="mt-2 text-[#666]">
                                Please speak naturally and try again.
                              </p>
                            )}
                          </div>
                        )}

                        {enrollment !== "recording" && enrollment !== "processing" && (
                          <div className="mt-5 space-y-3">
                            <button
                              type="button"
                              onClick={() => void startVoiceEnrollment()}
                              disabled={permission !== "granted" || disconnected || enrollmentDuration === null}
                              className="inline-flex min-h-[48px] w-full items-center justify-center rounded-full bg-[#635BFF] px-6 py-3 text-[14px] font-semibold text-white transition hover:-translate-y-[1px] disabled:cursor-not-allowed disabled:opacity-40"
                            >
                              {enrollment === "failed" ? "Record again" : "Start recording"}
                            </button>
                            <div className="text-center">
                              <button
                                type="button"
                                onClick={skipVoiceEnrollment}
                                className="min-h-[44px] px-4 py-2 text-[13px] font-semibold text-[#666] underline-offset-4 hover:text-[#111] hover:underline"
                              >
                                Continue without voice profile
                              </button>
                              <p className="mt-1 text-[12px] text-[#8A8A86]">Speaker identification may be less accurate.</p>
                            </div>
                          </div>
                        )}
                      </motion.section>
                    )}

                    {test === "passed" && voiceComplete && (
                      <div className="rounded-2xl border border-[#18A874]/30 bg-[#18A874]/10 p-4" role="status">
                        <div className="flex items-start gap-2">
                          <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-[#18A874]" />
                          <div>
                            <p className="text-[14px] font-semibold">Voice Profile · {enrollment === "ready" ? "Ready" : "Skipped"}</p>
                            {enrollment === "ready" && speechSeconds !== null && (
                              <>
                                <p className="mt-1 text-[13px] text-[#245C47]">Speech detected: {speechSeconds.toFixed(1)} seconds</p>
                                <p className="mt-1 text-[13px] text-[#245C47]">Voice profile is ready for speaker verification.</p>
                              </>
                            )}
                            {enrollment === "skipped" && <p className="mt-1 text-[13px] text-[#666]">Speaker identification may be less accurate.</p>}
                          </div>
                        </div>
                      </div>
                    )}

                    {test === "passed" && voiceComplete && (
                      <motion.section
                        aria-label="Where are you sitting?"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3 }}
                      >
                        <h2 className="text-[16px] font-bold tracking-[-0.01em]">Where are you sitting?</h2>
                        <p className="mt-1 text-[14px] text-[#666]">
                          An approximate position can help Roundtable understand which microphone is closest to each speaker.
                        </p>
                        <div className="mt-4">
                          <SeatPositionSelector selected={seat} onSelect={chooseSeat} onSkip={() => setSeatSkipped(true)} />
                        </div>
                      </motion.section>
                    )}

                    {disconnected ? (
                      <p className="rounded-2xl border border-[#E4E4E0] p-4 text-center text-[13px] text-[#666]">
                        Reconnect your microphone to continue.
                      </p>
                    ) : null}

                    <div className="space-y-3">
                      <div className="rounded-2xl border border-[#E4E4E0] p-4 text-[13px] text-[#555]">
                        <p className="font-semibold text-[#111]">Before you enter</p>
                        <ul className="mt-2 space-y-1.5">
                          <li>✓ Microphone connected</li>
                          <li>✓ Microphone tested</li>
                          <li>{enrollment === "ready" ? "✓ Voice profile ready" : "✓ Voice profile skipped"}</li>
                          <li>{seat || seatSkipped ? "✓ Position selected or skipped" : "○ Position optional"}</li>
                        </ul>
                      </div>
                    <button
                      type="button"
                      disabled={!ready}
                      onClick={continueToLiveSession}
                      className="group inline-flex min-h-[52px] w-full items-center justify-center gap-2 rounded-full bg-[#111111] px-8 py-4 text-[15px] font-semibold text-white transition-all hover:-translate-y-[1px] hover:shadow-xl disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      {ready ? (
                        <>
                          Ready — Enter Roundtable <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-[2px]" />
                        </>
                      ) : (
                        "Complete the microphone test to continue"
                      )}
                    </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          <p className="mt-5 text-center text-[12px] leading-relaxed text-[#8A8A86]">
            {participant ? (
              <>Signed in as {participant.displayName ?? "participant"} · temporary access for this session only.</>
            ) : (
              <>No account needed — access is temporary for this session only.</>
            )}
          </p>
          {ready && participant?.sessionId && (
            <div className="mt-4 text-center">
              <button
                type="button"
                onClick={continueToLiveSession}
                className="inline-flex min-h-[48px] items-center justify-center rounded-full border border-[#E4E4E0] bg-white px-6 py-3 text-[14px] font-semibold text-[#111111] transition hover:-translate-y-[1px] hover:border-[#111111]"
              >
                Join Live Session
              </button>
            </div>
          )}
        </motion.div>
      </main>
      <Footer />
    </div>
  );
}
