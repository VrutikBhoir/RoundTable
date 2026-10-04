import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Mic, MicOff, Radio, RefreshCw, UserCheck, UserRound, UserX, Users, Volume2, VolumeX, Wifi, WifiOff } from "lucide-react";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import AudioLevelMeter from "../components/mic/AudioLevelMeter";
import { approveJoinRequest, getLobby, getVoiceStatus, lockSession, rejectJoinRequest, unlockSession, wsBaseUrl, type JoinRequest } from "../lib/api";
import { apiBaseUrl } from "../lib/api";
import { useSessionAudio } from "../lib/useSessionAudio";
import { startTranscriptionFeed } from "../lib/transcriptionFeed";
import { enrollVoiceProfile } from "../lib/voiceEnrollment";
import { microphoneProcessingStatus } from "../lib/audio";
import type { ParticipantAudioStreamManager } from "../lib/participantAudio";

export type FusedTranscriptEntry = {
  id: string;
  start: number;
  end: number;
  speaker_id: string;
  text: string;
  confidence: number;
  source_participant_id: string;
  ambiguous: boolean;
  language?: string;
  isFinal?: boolean;
  quality?: number;
  overlap_participants?: string[];
};

type SessionRole = "host" | "participant";
type ConnectionState = "connecting" | "connected" | "disconnected" | "reconnecting";
type LiveParticipant = {
  id: string;
  display_name: string;
  role: SessionRole;
  connection_state: ConnectionState;
  microphone_state: "connected" | "muted" | "disconnected";
  last_seen_at?: number;
  last_audio_at?: number | null;
};

type SessionState = {
  sessionId?: string | null;
  hostToken?: string | null;
  participantId?: string | null;
  participantToken?: string | null;
  displayName?: string | null;
  sessionName?: string | null;
  role?: SessionRole;
};

type SessionEvent = {
  id: string;
  message: string;
  kind: "join" | "leave" | "info";
  at: number;
};

type PeerSignalMessage = {
  type: "peer_signal";
  from_id: string;
  signal_type: "offer" | "answer" | "candidate";
  signal: RTCSessionDescriptionInit | RTCIceCandidateInit;
};

function upsertJoinRequests(current: JoinRequest[], incoming: JoinRequest[]): JoinRequest[] {
  const byId = new Map(current.map((request) => [request.id, request]));
  incoming.forEach((request) => byId.set(request.id, request));
  return [...byId.values()];
}

function RemoteAudio({
  peerId,
  stream,
  muted,
  volume,
  audioElementsRef,
  onPlaybackBlocked,
}: {
  peerId: string;
  stream: MediaStream;
  muted: boolean;
  volume: number;
  audioElementsRef: React.MutableRefObject<Map<string, HTMLAudioElement>>;
  onPlaybackBlocked: (blocked: boolean) => void;
}) {
  const audioRef = useRef<HTMLAudioElement>(null);

  useLayoutEffect(() => {
    const element = audioRef.current;
    if (!element) return;
    element.muted = muted;
    element.volume = volume;
  }, [muted, volume]);

  useEffect(() => {
    const element = audioRef.current;
    if (!element) return;
    element.srcObject = stream;
    audioElementsRef.current.set(peerId, element);
    void element.play().then(() => onPlaybackBlocked(false)).catch(() => onPlaybackBlocked(true));
    return () => {
      audioElementsRef.current.delete(peerId);
      element.srcObject = null;
    };
  }, [peerId, stream, audioElementsRef, onPlaybackBlocked]);

  return <audio ref={audioRef} autoPlay playsInline className="sr-only" aria-label={`${peerId} live audio`} />;
}

/* Developer-only stream-identity diagnostic. Reads the central registry
 * (re-rendered via `version`) — never the playback elements. */
function StreamDiagnostics({
  manager,
  version,
  localParticipantId,
}: {
  manager: ParticipantAudioStreamManager;
  version: number;
  localParticipantId: string;
}) {
  void version;
  const ids = manager.participantIds();
  return (
    <section
      aria-label="Audio stream diagnostics"
      className="rounded-[24px] border border-dashed border-[#E4E4E0] bg-white p-6"
    >
      <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-[#8A8A86]">
        Stream diagnostics · {manager.size} stream{manager.size === 1 ? "" : "s"}
      </p>
      <div className="mt-4 space-y-2">
        {ids.length === 0 && (
          <p className="rounded-xl bg-[#F7F7F5] px-4 py-3 font-mono text-[12px] text-[#8A8A86]">
            No participant streams registered yet.
          </p>
        )}
        {ids.map((id) => {
          const stream = manager.getStream(id);
          const tracks = manager.getTracks(id);
          const liveTracks = tracks.filter((track) => track.readyState === "live").length;
          return (
            <div key={id} className="rounded-xl bg-[#F7F7F5] px-4 py-3 font-mono text-[12px] text-[#333]">
              <p>
                Participant: <span className="font-semibold text-[#111]">{id}</span>
                {id === localParticipantId && <span className="text-[#8A8A86]"> (local)</span>}
              </p>
              <p>Stream: {stream ? "available" : "missing"}</p>
              <p>Tracks: {tracks.length}</p>
              <p>Track kind: {tracks.map((track) => track.kind).join(", ") || "—"}</p>
              <p>
                Track state:{" "}
                {tracks.length === 0 ? "—" : `${liveTracks} live / ${tracks.length - liveTracks} ended`}
              </p>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function readStoredHostSession(): { sessionId: string; hostToken: string; sessionName: string } | null {
  try {
    const raw = sessionStorage.getItem("roundtable.hostSession");
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { sessionId?: string; hostToken?: string; sessionName?: string };
    if (!parsed.sessionId || !parsed.hostToken) return null;
    return { sessionId: parsed.sessionId, hostToken: parsed.hostToken, sessionName: parsed.sessionName ?? "Roundtable Session" };
  } catch {
    return null;
  }
}

function readStoredParticipantSession(): SessionState | null {
  try {
    const raw = sessionStorage.getItem("roundtable.participantSession");
    if (!raw) return null;
    const parsed = JSON.parse(raw) as SessionState;
    if (!parsed.sessionId) return null;
    return {
      sessionId: parsed.sessionId,
      participantId: parsed.participantId ?? null,
      participantToken: parsed.participantToken ?? null,
      displayName: parsed.displayName ?? "Participant",
      sessionName: parsed.sessionName ?? "Roundtable Session",
      role: "participant",
    };
  } catch {
    return null;
  }
}

function normalizeSessionState(value: unknown): SessionState | null {
  if (!value || typeof value !== "object") return null;
  const candidate = value as Record<string, unknown> & { session?: Record<string, unknown> };

  const direct = candidate.sessionId ?? candidate.session?.sessionId;
  if (!direct || typeof direct !== "string") {
    const nested = candidate.session;
    if (!nested) return null;
    const nestedSessionId = nested.sessionId;
    if (typeof nestedSessionId !== "string") return null;
    return {
      sessionId: nestedSessionId,
      hostToken: typeof nested.hostToken === "string" ? nested.hostToken : null,
      participantId: typeof nested.participantId === "string" ? nested.participantId : null,
      participantToken: typeof nested.participantToken === "string" ? nested.participantToken : null,
      displayName: typeof nested.displayName === "string" ? nested.displayName : typeof nested.hostName === "string" ? nested.hostName : "Participant",
      sessionName: typeof nested.sessionName === "string" ? nested.sessionName : typeof nested.name === "string" ? nested.name : "Roundtable Session",
      role: candidate.role === "host" || candidate.role === "participant" ? candidate.role : nested.role === "host" || nested.role === "participant" ? nested.role : "participant",
    };
  }

  return {
    sessionId: typeof direct === "string" ? direct : null,
    hostToken: typeof candidate.hostToken === "string" ? candidate.hostToken : null,
    participantId: typeof candidate.participantId === "string" ? candidate.participantId : null,
    participantToken: typeof candidate.participantToken === "string" ? candidate.participantToken : null,
    displayName: typeof candidate.displayName === "string" ? candidate.displayName : "Participant",
    sessionName: typeof candidate.sessionName === "string" ? candidate.sessionName : "Roundtable Session",
    role: candidate.role === "host" || candidate.role === "participant" ? candidate.role : "participant",
  };
}

function readableState(state: ConnectionState) {
  switch (state) {
    case "connecting":
      return "Connecting";
    case "connected":
      return "Connected";
    case "reconnecting":
      return "Reconnecting";
    default:
      return "Disconnected";
  }
}

export default function RoundtableSession() {
  const location = useLocation();
  const navigate = useNavigate();
  const routeState = normalizeSessionState(location.state);
  const initialSession = routeState ?? readStoredHostSession() ?? readStoredParticipantSession();

  const session = initialSession as SessionState | null;
  const sessionId = session?.sessionId ?? null;
  const role = session?.role ?? (readStoredHostSession() ? "host" : "participant");

  const [participants, setParticipants] = useState<LiveParticipant[]>([]);
  const [sessionEvents, setSessionEvents] = useState<SessionEvent[]>([]);
  const [transcriptEntries, setTranscriptEntries] = useState<FusedTranscriptEntry[]>([]);

  // Upsert by id: partial results update the same row in place instead of
  // appending duplicate lines.
  const mergeTranscriptEntries = (incoming: FusedTranscriptEntry[]) => {
    if (!incoming.length) return;
    setTranscriptEntries((current) => {
      const byId = new Map(current.map((e) => [e.id, e]));
      let changed = false;
      for (const e of incoming) {
        if (!e.id || typeof e.text !== "string" || !e.text.trim()) {
          continue;
        }
        if (import.meta.env.DEV) {
          console.debug("TRANSCRIPT_EVENT", {
            participant_id: e.source_participant_id,
            segment_id: e.id,
            text: e.text,
            language: e.language ?? "unknown",
            confidence: e.confidence,
            is_final: e.isFinal ?? true,
            source: "live",
          });
        }
        const prev = byId.get(e.id);
        if (!prev || prev.text !== e.text || prev.isFinal !== e.isFinal || prev.end !== e.end) {
          byId.set(e.id, e);
          changed = true;
        }
      }
      if (!changed) return current;
      return [...byId.values()]
        .sort((a, b) => a.start - b.start || a.end - b.end)
        .slice(-200);
    });
  };
  const [socketState, setSocketState] = useState<ConnectionState>("connecting");
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playbackBlocked, setPlaybackBlocked] = useState(false);
  const [pendingRequests, setPendingRequests] = useState<JoinRequest[]>([]);
  const [requestAction, setRequestAction] = useState<string | null>(null);
  const [mutedParticipants, setMutedParticipants] = useState<Set<string>>(() => new Set());
  const [participantVolumes, setParticipantVolumes] = useState<Record<string, number>>({});
  const [voiceState, setVoiceState] = useState<"Not enrolled" | "Recording" | "Processing" | "Verified" | "Unavailable">("Not enrolled");
  const [enrollmentProgress, setEnrollmentProgress] = useState(0);
  const [enrollmentSpeechSeconds, setEnrollmentSpeechSeconds] = useState<number | null>(null);
  const [enrollmentDuration, setEnrollmentDuration] = useState<number | null>(null);
  const [meetingLocked, setMeetingLocked] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const previousParticipantIdsRef = useRef<string[]>([]);
  const remoteAudioElementsRef = useRef(new Map<string, HTMLAudioElement>());
  const transcriptScrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setTranscriptEntries([]);
    setSessionEvents([]);
  }, [sessionId]);

  useEffect(() => {
    const container = transcriptScrollRef.current;
    if (container) container.scrollTop = container.scrollHeight;
  }, [transcriptEntries]);

  const localParticipantKey = role === "host" ? "host" : (session?.participantId ?? "participant");
  const audio = useSessionAudio(socket, participants, localParticipantKey);

  useEffect(() => {
    return () => {
      if (reconnectTimerRef.current) window.clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) {
        try {
          wsRef.current.send(JSON.stringify({ type: "leave" }));
        } catch {
          // Ignore leave send failures during cleanup.
        }
        try {
          wsRef.current.close();
        } catch {
          // Ignore close errors on cleanup.
        }
      }
      wsRef.current = null;
    };
  }, []);

  useEffect(() => {
    const currentIds = participants.map((participant) => participant.id);
    const previousIds = previousParticipantIdsRef.current;
    const joined = currentIds.filter((id) => !previousIds.includes(id));
    const left = previousIds.filter((id) => !currentIds.includes(id));

    if (joined.length || left.length) {
      const nextMessages: SessionEvent[] = [];
      joined.forEach((id) => {
        const participant = participants.find((p) => p.id === id);
        if (participant) {
          nextMessages.push({ id: `join-${id}`, message: `${participant.display_name} joined the session`, kind: "join", at: Date.now() });
        }
      });
      left.forEach((id) => {
        const participant = previousParticipantIdsRef.current.length > 0
          ? participants.find((p) => p.id === id) ?? { display_name: "A participant" }
          : { display_name: "A participant" };
        nextMessages.push({ id: `leave-${id}`, message: `${participant.display_name} left the session`, kind: "leave", at: Date.now() });
      });
      if (nextMessages.length) {
        setSessionEvents((existing) => {
          const byId = new Map(existing.map((event) => [event.id, event]));
          nextMessages.forEach((event) => byId.set(event.id, event));
          return [...byId.values()].sort((a, b) => b.at - a.at).slice(0, 5);
        });
      }
    }
    previousParticipantIdsRef.current = currentIds;
  }, [participants]);

  // Initial transcript snapshot once connected (covers entries made before join).
  useEffect(() => {
    if (socketState !== "connected" || !sessionId) return;
    let active = true;
    const params = new URLSearchParams();
    if (session?.hostToken) params.set("host_token", session.hostToken);
    if (session?.participantId && session?.participantToken) {
      params.set("participant_id", session.participantId);
      params.set("participant_token", session.participantToken);
    }
    void fetch(`${apiBaseUrl()}/api/sessions/${encodeURIComponent(sessionId)}/transcript?${params.toString()}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((body) => {
        if (active && body && Array.isArray(body.entries)) {
          mergeTranscriptEntries(body.entries as FusedTranscriptEntry[]);
        }
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [socketState, sessionId]);

  // Live feed: this device's own mic -> backend pipeline in ~4 s windows.
  // Independent of WebRTC transport; results return via transcript_event.
  useEffect(() => {
    const stream = audio.localStream;
    if (!stream || !sessionId || audio.audioState !== "connected") return;
    const credentials = session?.hostToken
      ? { host_token: session.hostToken }
      : session?.participantId && session?.participantToken
        ? { participant_id: session.participantId, participant_token: session.participantToken }
        : null;
    if (!credentials) return;
    const stop = startTranscriptionFeed({
      stream,
      sessionId,
      credentials,
      apiBase: apiBaseUrl(),
    });
    return () => {
      stop();
    };
  }, [audio.localStream, audio.audioState, sessionId, session?.hostToken, session?.participantId, session?.participantToken]);

  useEffect(() => {
    const activeIds = new Set(participants.map((participant) => participant.id));
    setMutedParticipants((current) => new Set([...current].filter((id) => activeIds.has(id))));
    setParticipantVolumes((current) => Object.fromEntries(
      Object.entries(current).filter(([id]) => activeIds.has(id))
    ));
  }, [participants]);

  useEffect(() => {
    if (role !== "host" || !session?.hostToken || !sessionId) {
      setPendingRequests([]);
      return;
    }

    let active = true;
    const refreshRequests = async () => {
      try {
        const lobby = await getLobby(sessionId, session.hostToken!);
        if (active) {
          setPendingRequests((current) => upsertJoinRequests(current, lobby.pending_requests));
          setMeetingLocked(lobby.status === "LOCKED");
        }
      } catch {
        if (active) setPendingRequests([]);
      }
    };

    void refreshRequests();
    const interval = window.setInterval(() => void refreshRequests(), 3000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [role, session?.hostToken, sessionId]);

  async function decideJoinRequest(request: JoinRequest, decision: "approve" | "deny") {
    if (!session?.hostToken || !sessionId) return;
    setRequestAction(request.id);
    setError(null);
    try {
      const lobby = decision === "approve"
        ? await approveJoinRequest(sessionId, request.id, session.hostToken)
        : await rejectJoinRequest(sessionId, request.id, session.hostToken);
      setPendingRequests((current) => upsertJoinRequests(current, lobby.pending_requests));
      const event: SessionEvent = {
        id: `${decision}-${request.id}-${Date.now()}`,
        message: decision === "approve" ? `${request.display_name} was approved to join` : `${request.display_name}'s request was declined`,
        kind: "info",
        at: Date.now(),
      };
      setSessionEvents((existing) => [event, ...existing].slice(0, 5));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not update the join request.");
    } finally {
      setRequestAction(null);
    }
  }

  function toggleParticipantMute(participantId: string) {
    setMutedParticipants((current) => {
      const next = new Set(current);
      if (next.has(participantId)) next.delete(participantId);
      else next.add(participantId);
      return next;
    });
  }

  useEffect(() => {
    if (!sessionId) return;
    let stopped = false;

    const connect = () => {
      if (stopped) return;
      // Direct to the FastAPI backend (wsBaseUrl); VITE_WS_URL /
      // VITE_API_URL override for remote. Never routed via Vite proxy.
      const base = wsBaseUrl();
      const params = new URLSearchParams();

      if (role === "host" && session?.hostToken) {
        params.set("host_token", session.hostToken);
      }
      if (role === "participant" && session?.participantId && session?.participantToken) {
        params.set("participant_id", session.participantId);
        params.set("participant_token", session.participantToken);
      }

      const ws = new WebSocket(`${base}/ws/sessions/${sessionId}${params.toString() ? `?${params.toString()}` : ""}`);
      wsRef.current = ws;

      ws.onopen = () => {
        if (stopped) {
          ws.close();
          return;
        }
        setError(null);
        setSocketState("connected");
        setSocket(ws);
        const liveEvent: SessionEvent = { id: `live-${Date.now()}`, message: "Realtime connection is live", kind: "info", at: Date.now() };
        setSessionEvents((existing) => [liveEvent, ...existing].slice(0, 5));
      };

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as {
            type?: string;
            participants?: LiveParticipant[];
            session_id?: string;
            from_id?: string;
            signal_type?: PeerSignalMessage["signal_type"];
            signal?: PeerSignalMessage["signal"];
          };

          if (payload.type === "session_state" && payload.participants) {
            setParticipants(payload.participants);
            return;
          }

          if (payload.type === "pong") {
            setSocketState("connected");
          }

          if (payload.type === "transcript_event" && Array.isArray((payload as { entries?: unknown }).entries)) {
            mergeTranscriptEntries((payload as { entries: FusedTranscriptEntry[] }).entries);
          }

          if (payload.type === "peer_signal" && payload.from_id && payload.signal_type && payload.signal) {
            audio.handleSignal(payload as PeerSignalMessage);
          }
        } catch {
          // Ignore malformed realtime messages and keep the session alive.
        }
      };

      ws.onclose = (event: CloseEvent) => {
        if (wsRef.current === ws) {
          wsRef.current = null;
          setSocket(null);
        }
        if (stopped) return;
        // Fatal closes must not retry forever: the session is gone (4404,
        // e.g. backend restarted and dropped its in-memory store) or the
        // credentials are invalid (4401). Anything else is transient.
        if (event.code === 4404) {
          setSocketState("disconnected");
          setError("This Roundtable session no longer exists on the server. Create a new session or rejoin with a fresh invitation.");
          return;
        }
        if (event.code === 4401) {
          setSocketState("disconnected");
          setError("Your access to this session is no longer valid. Rejoin with a fresh invitation.");
          return;
        }
        setSocketState("reconnecting");
        setError("Connection lost. Reconnecting to the Roundtable session...");
        if (reconnectTimerRef.current) window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = window.setTimeout(() => {
          connect();
        }, 1500);
      };

      ws.onerror = () => {
        setSocketState("disconnected");
        setError("Realtime connection failed. Retrying...");
      };
    };

    connect();

    const heartbeat = window.setInterval(() => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "heartbeat" }));
      }
    }, 4000);

    const handleBeforeUnload = () => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "leave" }));
      }
    };
    window.addEventListener("beforeunload", handleBeforeUnload);

    return () => {
      window.clearInterval(heartbeat);
      window.removeEventListener("beforeunload", handleBeforeUnload);
      if (reconnectTimerRef.current) window.clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) {
        stopped = true;
        try {
          wsRef.current.send(JSON.stringify({ type: "leave" }));
        } catch {
          // Ignore leave send errors during cleanup.
        }
        try {
          wsRef.current.close();
        } catch {
          // Ignore close errors in cleanup.
        }
      }
      stopped = true;
      setSocket(null);
      wsRef.current = null;
    };
  }, [sessionId, role, session?.hostToken, session?.participantId, session?.participantToken, session?.displayName]);

  if (!sessionId) return <Navigate to="/" replace />;

  const sessionName = session?.sessionName ?? "Roundtable Session";
  const uniqueParticipants = Array.from(
    new Map(participants.map((participant) => [`${participant.role}:${participant.id}`, participant])).values()
  );
  const visibleParticipants = uniqueParticipants.length > 0
    ? uniqueParticipants
    : [
        {
          id: localParticipantKey,
          display_name: session?.displayName ?? (role === "host" ? "Host" : "Participant"),
          role,
          connection_state: socketState,
          microphone_state: audio.audioState === "connected" ? audio.micMuted ? "muted" : "connected" : "disconnected",
          last_audio_at: null,
        },
      ];

  const sessionStatusText = socketState !== "connected"
    ? socketState === "reconnecting" ? "Mic reconnecting" : "Mic offline"
    : audio.micMuted
      ? "Mic muted"
      : audio.audioState === "connected"
        ? "Mic live"
        : "Mic offline";
  const connectedAudioPeers = Object.values(audio.peerStates).filter((state) => state === "connected").length;
  const localNoiseSuppression = audio.localStream ? microphoneProcessingStatus(audio.localStream) : "UNAVAILABLE";
  const localCredentials: Record<string, string> | null = session?.hostToken
    ? { host_token: session.hostToken }
    : session?.participantId && session?.participantToken
      ? { participant_id: session.participantId, participant_token: session.participantToken }
      : null;

  useEffect(() => {
    if (!sessionId || !localCredentials) return;
    let active = true;
    void getVoiceStatus(sessionId, localCredentials).then((status) => {
      console.debug("[voice] live profile lookup", {
        session_id: sessionId,
        participant_id: status.participant_id,
        status: status.status,
      });
      if (!active) return;
      setEnrollmentDuration(status.enrollment_duration_seconds);
      if (status.status === "ready") setVoiceState("Verified");
      else if (status.status !== "unavailable") setVoiceState("Not enrolled");
    }).catch((lookupError) => {
      console.warn("[voice] live profile lookup failed", {
        session_id: sessionId,
        participant_id: session?.participantId ?? "host",
        error: lookupError instanceof Error ? lookupError.message : lookupError,
      });
    });
    return () => {
      active = false;
    };
  }, [sessionId, session?.hostToken, session?.participantId, session?.participantToken]);

  const startEnrollment = async () => {
    if (!audio.localStream || !sessionId || !localCredentials) return;
    setVoiceState("Recording");
    setEnrollmentProgress(0);
    try {
      setVoiceState("Recording");
      const result = await enrollVoiceProfile({
        stream: audio.localStream,
        endpoint: `${apiBaseUrl()}/api/sessions/${encodeURIComponent(sessionId)}/voice-enrollment`,
        credentials: localCredentials,
        onProgress: (seconds) => setEnrollmentProgress(seconds),
        seconds: enrollmentDuration ?? 0,
        onProcessing: () => setVoiceState("Processing"),
      });
      console.debug("[voice] live enrollment response", {
        session_id: sessionId,
        participant_id: result.participantId,
        speech_seconds: result.speechSeconds,
        recording_seconds: result.recordingSeconds,
      });
      setEnrollmentSpeechSeconds(result.speechSeconds);
      setVoiceState("Verified");
    } catch (enrollmentError) {
      setVoiceState("Unavailable");
      setError(enrollmentError instanceof Error ? enrollmentError.message : "Voice enrollment failed.");
    }
  };

  const enableRemoteAudio = () => {
    void Promise.all([...remoteAudioElementsRef.current.values()].map((element) => element.play()))
      .then(() => setPlaybackBlocked(false))
      .catch(() => setPlaybackBlocked(true));
  };

  return (
    <div className="min-h-screen bg-[#F7F7F5] text-[#111111]">
      <Navbar />
      <main className="mx-auto max-w-[980px] px-5 pb-24 pt-[118px] md:px-8 md:pt-[150px]">
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45 }}>
          <div className="mb-6 flex items-center justify-between gap-3">
            <p className="inline-flex items-center gap-2 rounded-full border border-[#E4E4E0] bg-white px-4 py-1.5 font-mono text-[11px] uppercase tracking-[0.16em] text-[#666]">
              <span className="h-[7px] w-[7px] rounded-full bg-[#18A874]" /> Roundtable Session
            </p>
            <div className="inline-flex items-center gap-2 rounded-full border border-[#E4E4E0] bg-white px-3 py-2 text-[12px] font-semibold text-[#666]">
              {socketState === "connected" ? <Wifi className="h-3.5 w-3.5 text-[#18A874]" /> : <WifiOff className="h-3.5 w-3.5 text-[#8A8A86]" />}
              {readableState(socketState)}
            </div>
          </div>

          <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
            <section className="rounded-[24px] border border-[#E4E4E0] bg-white p-6 shadow-[0_16px_48px_rgba(0,0,0,0.04)] sm:p-8">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-[#8A8A86]">{sessionName}</p>
                  <h1 className="editorial-tight mt-3 text-[38px] font-[700] sm:text-[52px]">Participants</h1>
                  <p className="mt-2 text-[13px] text-[#666]">{visibleParticipants.length} {visibleParticipants.length === 1 ? "device" : "devices"} connected</p>
                </div>
                <div className="grid h-12 w-12 place-items-center rounded-full bg-[#EAE8FF] text-[#635BFF]">
                  <Users className="h-5 w-5" />
                </div>
              </div>

              <div className="mt-8 space-y-3">
                {visibleParticipants.map((participant) => {
                  const isLocal = participant.id === localParticipantKey;
                  const isAudioActive = audio.speakingPeerIds.includes(participant.id);
                  const isMutedForYou = mutedParticipants.has(participant.id);
                  const participantVolume = participantVolumes[participant.id] ?? 1;
                  const peerState = audio.peerStates[participant.id];
                  const participantState: ConnectionState = isLocal
                    ? participant.connection_state
                    : peerState === "connected"
                      ? "connected"
                      : peerState === "failed" || peerState === "closed"
                        ? "disconnected"
                        : peerState === "disconnected"
                          ? "reconnecting"
                          : "connecting";
                  const badgeClass = participantState === "connected"
                    ? "bg-[#EAF9F2] text-[#18A874]"
                    : participantState === "reconnecting"
                      ? "bg-[#FDF6E3] text-[#9A7B1F]"
                      : participantState === "connecting"
                        ? "bg-[#EAE8FF] text-[#635BFF]"
                        : "bg-[#F2F2F1] text-[#666]";

                  return (
                    <div key={`${participant.id}-${participant.display_name}`} className="rounded-2xl bg-[#F7F7F5] px-4 py-3">
                      <div className="flex items-center justify-between gap-3">
                        <div className="flex items-center gap-3">
                          <span className="grid h-9 w-9 place-items-center rounded-full bg-[#EAE8FF] text-[12px] font-bold text-[#635BFF]">
                            {participant.display_name?.charAt(0)?.toUpperCase() ?? "R"}
                          </span>
                          <div>
                            <p className="text-[15px] font-semibold">{participant.display_name}</p>
                            <p className="text-[12px] text-[#666]">{participant.role === "host" ? "Host" : "Participant"}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className={`inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-semibold ${badgeClass}`}>
                            {readableState(participantState)}
                          </span>
                          {!isLocal && (
                            <button
                              type="button"
                              onClick={() => toggleParticipantMute(participant.id)}
                              aria-label={`${isMutedForYou ? "Unmute" : "Mute"} ${participant.display_name} for you`}
                              title={`${isMutedForYou ? "Unmute" : "Mute"} ${participant.display_name} for you`}
                              className="grid h-9 w-9 place-items-center rounded-full border border-[#E4E4E0] bg-white text-[#555] transition hover:border-[#111] hover:text-[#111]"
                            >
                              {isMutedForYou ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
                            </button>
                          )}
                        </div>
                      </div>

                      <div className="mt-3">
                        <div className="mb-1 flex items-center justify-between text-[10px] font-medium uppercase tracking-[0.12em] text-[#8A8A86]">
                          <span>Mic</span>
                          <span>{isLocal ? audio.localSpeaking ? "Speaking" : sessionStatusText : isMutedForYou ? "Muted for you" : participant.microphone_state === "muted" ? "Mic muted" : isAudioActive ? "Speaking" : "Listening"}</span>
                        </div>
                        {isLocal ? (
                          <AudioLevelMeter getLevel={() => audio.localLevelRef.current} live={audio.audioState === "connected" && !audio.micMuted} label="Local mic activity" />
                        ) : (
                          <AudioLevelMeter
                            getLevel={() => audio.remoteLevelsRef.current.get(participant.id) ?? 0}
                            live={audio.peerStates[participant.id] === "connected" && participant.microphone_state !== "muted"}
                            label={`${participant.display_name} mic activity`}
                          />
                        )}
                      </div>
                      {!isLocal && (
                        <div className="mt-3 flex items-center gap-3 border-t border-[#E4E4E0] pt-3">
                          <Volume2 className={`h-4 w-4 shrink-0 ${isMutedForYou ? "text-[#8A8A86]" : "text-[#635BFF]"}`} />
                          <label className="sr-only" htmlFor={`participant-volume-${participant.id}`}>
                            Listening volume for {participant.display_name}
                          </label>
                          <input
                            id={`participant-volume-${participant.id}`}
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={participantVolume}
                            onChange={(event) => setParticipantVolumes((current) => ({
                              ...current,
                              [participant.id]: Number(event.target.value),
                            }))}
                            aria-label={`Listening volume for ${participant.display_name}`}
                            aria-valuetext={`${Math.round(participantVolume * 100)} percent`}
                            className="h-2 min-w-0 flex-1 cursor-pointer accent-[#635BFF]"
                          />
                          <span className="w-10 text-right font-mono text-[11px] tabular-nums text-[#666]">
                            {Math.round(participantVolume * 100)}%
                          </span>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </section>

            <aside className="space-y-5">
              <section className="rounded-[24px] border border-[#E4E4E0] bg-white p-6 shadow-[0_16px_48px_rgba(0,0,0,0.04)]">
                <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-[#8A8A86]">Live status</p>
                <div className="mt-4 rounded-2xl bg-[#F7F7F5] p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2 text-[16px] font-semibold">
                      <Radio className="h-4 w-4 text-[#635BFF]" />
                      Connection
                    </div>
                    <span className="text-[12px] font-semibold uppercase tracking-[0.08em] text-[#666]">{readableState(socketState)}</span>
                  </div>
                  <div className="mt-4 flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2 text-[16px] font-semibold">
                      <Mic className="h-4 w-4 text-[#18A874]" />
                      Microphone
                    </div>
                    <span className="text-[12px] font-semibold uppercase tracking-[0.08em] text-[#666]">{sessionStatusText}</span>
                  </div>
                  <div className="mt-4">
                    <AudioLevelMeter getLevel={() => audio.localLevelRef.current} live={audio.audioState === "connected" && !audio.micMuted} label="Your mic activity" />
                  </div>
                  <div className="mt-4 flex flex-col gap-2 sm:flex-row">
                    <button
                      type="button"
                      onClick={audio.toggleMuted}
                      disabled={audio.audioState !== "connected"}
                      className="inline-flex min-h-11 items-center justify-center gap-2 rounded-full border border-[#E4E4E0] bg-white px-4 py-2 text-[13px] font-semibold transition hover:border-[#111] disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {audio.micMuted ? <Mic className="h-4 w-4" /> : <MicOff className="h-4 w-4" />}
                      {audio.micMuted ? "Unmute" : "Mute mic"}
                    </button>
                    <label className="sr-only" htmlFor="live-microphone-input">Microphone input</label>
                    <select
                      id="live-microphone-input"
                      aria-label="Microphone input"
                      value={audio.selectedDeviceId}
                      onChange={(event) => void audio.switchInput(event.target.value)}
                      disabled={audio.audioState !== "connected" || audio.inputDevices.length === 0}
                      className="min-h-11 min-w-0 flex-1 rounded-full border border-[#E4E4E0] bg-white px-4 py-2 text-[13px] font-medium text-[#333] disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {audio.inputDevices.length === 0 ? <option value="">Microphone input</option> : audio.inputDevices.map((device, index) => (
                        <option key={device.deviceId} value={device.deviceId}>{device.label || `Microphone ${index + 1}`}</option>
                      ))}
                    </select>
                  </div>
                  {audio.audioState === "disconnected" && socketState === "connected" && (
                    <button
                      type="button"
                      onClick={audio.retryMicrophone}
                      className="mt-3 inline-flex min-h-10 items-center justify-center gap-2 rounded-full border border-[#E4E4E0] bg-white px-4 py-2 text-[13px] font-semibold hover:border-[#111]"
                    >
                      <RefreshCw className="h-4 w-4" /> Retry microphone
                    </button>
                  )}
                  <p className="mt-3 text-[12px] text-[#8A8A86]">WebRTC audio links: {connectedAudioPeers} / {Math.max(0, visibleParticipants.length - 1)}</p>
                  {playbackBlocked && (
                    <button
                      type="button"
                      onClick={enableRemoteAudio}
                      className="mt-3 inline-flex min-h-10 w-full items-center justify-center gap-2 rounded-full bg-[#111] px-4 py-2 text-[13px] font-semibold text-white"
                    >
                      <Volume2 className="h-4 w-4" /> Enable participant audio
                    </button>
                  )}
                  {Object.entries(audio.remoteStreams).map(([peerId, stream]) => (
                    <RemoteAudio
                      key={peerId}
                      peerId={peerId}
                      stream={stream}
                      muted={mutedParticipants.has(peerId)}
                      volume={participantVolumes[peerId] ?? 1}
                      audioElementsRef={remoteAudioElementsRef}
                      onPlaybackBlocked={setPlaybackBlocked}
                    />
                  ))}
                </div>
              </section>

              <section aria-label="Live transcript" className="rounded-[24px] border border-[#E4E4E0] bg-white p-6 shadow-[0_16px_48px_rgba(0,0,0,0.04)]">
                <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-[#8A8A86]">Live transcript</p>
                <div ref={transcriptScrollRef} className="mt-4 max-h-72 space-y-2 overflow-y-auto" role="log" aria-label="Live transcript">
                  {transcriptEntries.length === 0 && (
                    <p className="rounded-xl border border-dashed border-[#E4E4E0] px-4 py-4 text-[13px] text-[#8A8A86]">
                      Captions will appear here as people speak.
                    </p>
                  )}
                  {transcriptEntries.map((entry) => {
                    const overlapNames = (entry.overlap_participants ?? [])
                      .map((id) => participants.find((p) => p.id === id)?.display_name ?? id)
                      .join(" + ");
                    const speakerName = entry.speaker_id === "unknown"
                      ? "Unknown"
                      : entry.speaker_id === "multiple"
                        ? overlapNames || "Multiple speakers"
                        : participants.find((p) => p.id === entry.speaker_id)?.display_name
                          ?? (entry.speaker_id === localParticipantKey ? (session?.displayName ?? "You") : "Participant");
                    const mm = Math.floor(entry.start / 60).toString().padStart(2, "0");
                    const ss = Math.floor(entry.start % 60).toString().padStart(2, "0");
                    const showTranscriptText = !entry.ambiguous || entry.confidence >= 0.3;
                    return (
                      <div key={entry.id} className="rounded-2xl bg-[#F7F7F5] px-4 py-3">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[#635BFF]">
                          [{mm}:{ss}] {speakerName}
                          {entry.isFinal === false && (
                            <span className="ml-1.5 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-[#635BFF]" aria-label="Updating" />
                          )}
                        </p>
                        {showTranscriptText ? (
                          <p className="mt-1 text-[14px] leading-relaxed text-[#111]">“{entry.text}”</p>
                        ) : (
                          <p className="mt-1 text-[14px] leading-relaxed text-[#8A8A86]">
                            ⚠ Overlapping / uncertain speech
                          </p>
                        )}
                        <p className="mt-1 text-[10px] uppercase tracking-[0.1em] text-[#8A8A86]">
                          {entry.language && entry.language !== "unknown" ? entry.language : "language uncertain"}
                          {entry.ambiguous && " · overlapping / uncertain"}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </section>

              <section aria-label="Voice profile" className="rounded-[24px] border border-[#E4E4E0] bg-white p-6 shadow-[0_16px_48px_rgba(0,0,0,0.04)]">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-[#8A8A86]">Voice profile</p>
                    <p className="mt-2 text-[15px] font-semibold">
                      {voiceState === "Verified" ? "Voice profile ready" : voiceState}
                    </p>
                    {voiceState === "Verified" && enrollmentSpeechSeconds !== null && (
                      <p className="mt-1 text-[12px] text-[#666]">
                        Speech detected: {enrollmentSpeechSeconds.toFixed(1)}s / 30s
                      </p>
                    )}
                    {voiceState === "Recording" && (
                      <p className="mt-1 text-[12px] text-[#666]">Speak naturally: {Math.floor(enrollmentProgress)} / 30 seconds</p>
                    )}
                  </div>
                  <button
                    type="button"
                    onClick={() => void startEnrollment()}
                    disabled={!audio.localStream || enrollmentDuration === null || voiceState === "Recording" || voiceState === "Processing"}
                    className="rounded-full bg-[#111] px-4 py-2 text-[12px] font-semibold text-white disabled:opacity-40"
                  >
                    {voiceState === "Verified" ? "Record again" : "Enroll voice"}
                  </button>
                </div>
                <p className="mt-3 text-[12px] text-[#666]">Noise suppression: {localNoiseSuppression}</p>
              </section>

              <StreamDiagnostics
                manager={audio.streamManager}
                version={audio.streamVersion}
                localParticipantId={localParticipantKey}
              />

              {role === "host" && (
                <section className="rounded-[24px] border border-[#E4E4E0] bg-white p-6 shadow-[0_16px_48px_rgba(0,0,0,0.04)]">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-[#8A8A86]">Host controls</p>
                      <h2 className="mt-2 text-[18px] font-bold">Join requests</h2>
                    </div>                    <span className="grid h-8 min-w-8 place-items-center rounded-full bg-[#F7F7F5] px-2 font-mono text-[12px]">{pendingRequests.length}</span>
                  </div>
                  <div className="mt-4 flex items-center justify-between gap-3 rounded-xl bg-[#F7F7F5] px-4 py-3">
                    <div>
                      <p className="text-[13px] font-semibold">Late joins</p>
                      <p className="text-[12px] text-[#666]">{meetingLocked ? "Meeting locked" : "Accepting requests"}</p>
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        if (!session?.hostToken || !sessionId) return;
                        const action = meetingLocked
                          ? unlockSession(sessionId, session.hostToken)
                          : lockSession(sessionId, session.hostToken);
                        void action.then((lobby) => {
                          setMeetingLocked(lobby.status === "LOCKED");
                          setPendingRequests(upsertJoinRequests([], lobby.pending_requests));
                        }).catch((actionError) => {
                          setError(actionError instanceof Error ? actionError.message : "Could not update meeting lock.");
                        });
                      }}
                      className="inline-flex min-h-9 items-center gap-1.5 rounded-full border border-[#E4E4E0] bg-white px-3 py-2 text-[12px] font-semibold hover:border-[#111]"
                    >
                      {meetingLocked ? "Unlock" : "Lock meeting"}
                    </button>
                  </div>
                  <div className="mt-4 space-y-2">
                    {pendingRequests.length === 0 ? (
                      <p className="rounded-xl border border-dashed border-[#E4E4E0] px-4 py-4 text-[13px] text-[#8A8A86]">No requests waiting for approval.</p>
                    ) : pendingRequests.map((request) => (
                      <div key={request.id} className="flex items-center justify-between gap-3 rounded-xl bg-[#F7F7F5] px-4 py-3">
                        <div className="min-w-0">
                          <p className="truncate text-[14px] font-semibold">{request.display_name}</p>
                          <p className="text-[12px] text-[#666]">Waiting to join</p>
                        </div>
                        <div className="flex shrink-0 items-center gap-2">
                          <button
                            type="button"
                            onClick={() => void decideJoinRequest(request, "approve")}
                            disabled={requestAction !== null}
                            aria-label={`Approve ${request.display_name}`}
                            title="Approve"
                            className="grid h-9 w-9 place-items-center rounded-full bg-[#EAF9F2] text-[#14895E] transition hover:bg-[#D9F3E7] disabled:opacity-50"
                          >
                            <UserCheck className="h-4 w-4" />
                          </button>
                          <button
                            type="button"
                            onClick={() => void decideJoinRequest(request, "deny")}
                            disabled={requestAction !== null}
                            aria-label={`Deny ${request.display_name}`}
                            title="Deny"
                            className="grid h-9 w-9 place-items-center rounded-full border border-[#E4E4E0] bg-white text-[#666] transition hover:border-[#B42318] hover:text-[#B42318] disabled:opacity-50"
                          >
                            <UserX className="h-4 w-4" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              <section className="rounded-[24px] border border-[#E4E4E0] bg-white p-6 shadow-[0_16px_48px_rgba(0,0,0,0.04)]">
                <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-[#8A8A86]">Session activity</p>
                <div className="mt-4 space-y-2">
                  {sessionEvents.length === 0 ? (
                    <div className="rounded-2xl border border-[#E4E4E0] bg-[#F7F7F5] px-4 py-3 text-[13px] text-[#666]">Waiting for live session activity…</div>
                  ) : (
                    sessionEvents.map((event) => (
                      <div key={event.id} className="rounded-2xl border border-[#E4E4E0] bg-[#F7F7F5] px-4 py-3 text-[13px] text-[#666]">
                        <span className={`mr-2 inline-block h-2 w-2 rounded-full ${event.kind === "join" ? "bg-[#18A874]" : event.kind === "leave" ? "bg-[#D97706]" : "bg-[#635BFF]"}`} />
                        {event.message}
                      </div>
                    ))
                  )}
                </div>
              </section>

              {(error || audio.audioError) && (
                <div role="alert" className="rounded-[20px] border border-[#E8D9A8] bg-[#FDF6E3] p-4 text-[14px] text-[#665B35]">
                  {error ?? audio.audioError}
                </div>
              )}

              <button
                type="button"
                onClick={() => navigate("/")}
                className="inline-flex min-h-[48px] w-full items-center justify-center rounded-full border border-[#E4E4E0] bg-white px-6 py-3 text-[14px] font-semibold text-[#111111] transition hover:-translate-y-[1px] hover:border-[#111111]"
              >
                Leave session
              </button>
            </aside>
          </div>
        </motion.div>
      </main>
      <Footer />
    </div>
  );
}
