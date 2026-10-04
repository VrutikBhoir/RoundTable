export type SessionStatus = "WAITING" | "LOCKED" | "FULL" | "STARTING" | "STARTED" | "ENDED";
export type RequestStatus = "pending" | "approved" | "rejected";

export type Invitation = {
  token: string;
  invitation_url: string;
  expires_at: string;
  active: boolean;
};

export type Participant = {
  id: string;
  display_name: string;
  status: "approved";
  created_at: string;
};

export type JoinRequest = {
  id: string;
  display_name: string;
  status: RequestStatus;
  created_at: string;
};

export type InvitationPreview = {
  session_id: string;
  session_name: string;
  host_name: string;
  capacity: number;
  participant_count: number;
  session_status: "open" | "in_progress" | "full" | "locked" | "started" | "ended";
  invitation: Invitation;
};

export type JoinRequestStatus = {
  id: string;
  display_name: string;
  status: RequestStatus;
  created_at: string;
  participant_id: string | null;
  participant_token: string | null;
};

export type Lobby = {
  session_id: string;
  name: string;
  capacity: number;
  status: SessionStatus;
  invitation: Invitation | null;
  participants: Participant[];
  pending_requests: JoinRequest[];
};

export type CreatedSession = {
  session_id: string;
  host_token: string;
  name: string;
  capacity: number;
  status: SessionStatus;
};

export type ApiError = Error & { code?: string; status?: number };

const API_BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

/** Effective HTTP backend base. Empty means same-origin (Vite proxies /api). */
export function apiBaseUrl(): string {
  return API_BASE;
}

/** WebSocket backend base. Order: explicit VITE_WS_URL, then the HTTP API
 *  base with its scheme swapped (https -> wss), else a DIRECT backend
 *  connection in local development (ws://hostname:8000).
 *
 *  Signaling deliberately bypasses the Vite dev proxy: proxying adds a
 *  middleman TCP hop whose aborted sockets surface as
 *  "[vite] ws proxy socket error: ECONNABORTED" whenever the page
 *  refreshes, StrictMode remounts, or a socket is replaced. The proxy
 *  entry stays only as a fallback for https pages without backend TLS. */
export function wsBaseUrl(): string {
  const override = ((import.meta.env.VITE_WS_URL as string | undefined) ?? "").replace(/\/$/, "");
  if (override) return override;
  if (API_BASE) return API_BASE.replace(/^http/, "ws");
  if (typeof window !== "undefined" && window.location.protocol === "https:") {
    return `wss://${window.location.host}`;
  }
  return `ws://${typeof window !== "undefined" ? window.location.hostname : "127.0.0.1"}:8000`;
}

async function request<T>(path: string, options: RequestInit = {}, hostToken?: string): Promise<T> {
  const headers = new Headers(options.headers);
  headers.set("Content-Type", "application/json");
  if (hostToken) headers.set("X-Host-Token", hostToken);

  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (!response.ok) {
    let code: string | undefined;
    let message = `Request failed (${response.status}).`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
      if (body.detail && typeof body.detail === "object") {
        code = body.detail.code;
        message = body.detail.message ?? message;
      }
    } catch {
      // Keep the HTTP status message when the server response is not JSON.
    }
    const error = new Error(message) as ApiError;
    error.code = code;
    error.status = response.status;
    throw error;
  }
  return response.json() as Promise<T>;
}

export function createSession(name: string, capacity: number) {
  return request<CreatedSession>("/api/sessions", { method: "POST", body: JSON.stringify({ name, capacity }) });
}

export function getLobby(sessionId: string, hostToken: string) {
  return request<Lobby>(`/api/sessions/${encodeURIComponent(sessionId)}/lobby`, {}, hostToken);
}

export function createInvitation(sessionId: string, hostToken: string) {
  return request<Invitation>(`/api/sessions/${encodeURIComponent(sessionId)}/invitation`, { method: "POST" }, hostToken);
}

export function regenerateInvitation(sessionId: string, hostToken: string) {
  return request<Invitation>(`/api/sessions/${encodeURIComponent(sessionId)}/invitation/regenerate`, { method: "POST" }, hostToken);
}

export function approveJoinRequest(sessionId: string, requestId: string, hostToken: string) {
  return request<Lobby>(`/api/sessions/${encodeURIComponent(sessionId)}/requests/${encodeURIComponent(requestId)}/approve`, { method: "POST" }, hostToken);
}

export function rejectJoinRequest(sessionId: string, requestId: string, hostToken: string) {
  return request<Lobby>(`/api/sessions/${encodeURIComponent(sessionId)}/requests/${encodeURIComponent(requestId)}/reject`, { method: "POST" }, hostToken);
}

export function removeParticipant(sessionId: string, participantId: string, hostToken: string) {
  return request<Lobby>(`/api/sessions/${encodeURIComponent(sessionId)}/participants/${encodeURIComponent(participantId)}`, { method: "DELETE" }, hostToken);
}

export function lockSession(sessionId: string, hostToken: string) {
  return request<Lobby>(`/api/sessions/${encodeURIComponent(sessionId)}/lock`, { method: "POST" }, hostToken);
}

export function unlockSession(sessionId: string, hostToken: string) {
  return request<Lobby>(`/api/sessions/${encodeURIComponent(sessionId)}/unlock`, { method: "POST" }, hostToken);
}

export function startSession(sessionId: string, hostToken: string) {
  return request<Lobby>(`/api/sessions/${encodeURIComponent(sessionId)}/start`, { method: "POST" }, hostToken);
}

export function requestToJoin(token: string, displayName: string) {
  return request<JoinRequest>(`/api/invitations/${encodeURIComponent(token)}/join`, { method: "POST", body: JSON.stringify({ display_name: displayName }) });
}

export function getInvitationPreview(token: string) {
  return request<InvitationPreview>(`/api/invitations/${encodeURIComponent(token)}`);
}

export function getJoinRequestStatus(token: string, requestId: string) {
  return request<JoinRequestStatus>(`/api/invitations/${encodeURIComponent(token)}/requests/${encodeURIComponent(requestId)}`);
}

export function getVoiceStatus(
  sessionId: string,
  credentials: { host_token?: string; participant_id?: string; participant_token?: string },
) {
  const params = new URLSearchParams();
  Object.entries(credentials).forEach(([key, value]) => {
    if (value) params.set(key, value);
  });
  return request<{ participant_id: string; status: string; enrollment_duration_seconds: number }>(
    `/api/sessions/${encodeURIComponent(sessionId)}/voice-status?${params.toString()}`,
  );
}
