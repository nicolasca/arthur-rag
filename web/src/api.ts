import type { AskResponse, Citation, RetrievedPassage } from "./types";

const ASK_ENDPOINT = "/api/ask";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isCitation(value: unknown): value is Citation {
  if (!isRecord(value)) return false;
  return (
    typeof value.evidence_id === "string" &&
    typeof value.chunk_id === "string" &&
    typeof value.quote === "string" &&
    typeof value.chapter_number === "number" &&
    typeof value.chapter_title === "string" &&
    typeof value.source_url === "string"
  );
}

function isRetrievedPassage(value: unknown): value is RetrievedPassage {
  if (!isRecord(value)) return false;
  return (
    typeof value.rank === "number" &&
    typeof value.chunk_id === "string" &&
    typeof value.similarity === "number" &&
    Number.isFinite(value.similarity) &&
    typeof value.chapter_number === "number" &&
    typeof value.chapter_title === "string" &&
    typeof value.source_url === "string" &&
    typeof value.text === "string"
  );
}

function isAskResponse(value: unknown): value is AskResponse {
  if (!isRecord(value)) return false;
  return (
    (value.status === "answered" || value.status === "insufficient") &&
    typeof value.answer === "string" &&
    Array.isArray(value.citations) &&
    value.citations.every(isCitation) &&
    Array.isArray(value.retrieved_passages) &&
    value.retrieved_passages.every(isRetrievedPassage)
  );
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    if (!response.ok) return null;
    throw new Error("Le serveur a renvoyé une réponse illisible.");
  }
}

function errorForStatus(status: number, payload: unknown): string {
  if (status === 422) {
    return isRecord(payload) && typeof payload.detail === "string"
      ? payload.detail
      : "La question n’est pas valide.";
  }
  if (status === 502) {
    return "Gemini n’a pas pu produire une réponse ancrée dans le corpus.";
  }
  if (status === 503) {
    return "Le service local n’est pas prêt. Vérifiez l’index et la configuration Gemini.";
  }
  return "La consultation a échoué côté serveur. Réessayez dans un instant.";
}

export async function askQuestion(question: string): Promise<AskResponse> {
  let response: Response;
  try {
    response = await fetch(ASK_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  } catch {
    throw new Error(
      "Impossible de joindre l’API locale. Vérifiez que FastAPI est démarré."
    );
  }

  const payload = await readJson(response);
  if (!response.ok) {
    throw new Error(errorForStatus(response.status, payload));
  }
  if (!isAskResponse(payload)) {
    throw new Error("Le serveur a renvoyé une réponse inattendue.");
  }
  return payload;
}
