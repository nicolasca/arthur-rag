import "@testing-library/jest-dom/vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { AskResponse } from "./types";

const SOURCE_URL =
  "https://fr.wikisource.org/wiki/Les_Enfances_de_Lancelot/07";

function passages() {
  return Array.from({ length: 5 }, (_, index) => ({
    rank: index + 1,
    chunk_id: `lancelot-07-chunk-00${index + 1}`,
    similarity: 0.89123 - index * 0.01,
    chapter_number: 7,
    chapter_title: "La Dame du Lac et Lancelot",
    source_url: SOURCE_URL,
    text: `Passage retrouvé numéro ${index + 1}.`,
  }));
}

const answeredResponse: AskResponse = {
  status: "answered",
  answer: "La Dame du Lac recueille Lancelot et veille à son éducation.",
  citations: [
    {
      evidence_id: "evidence-01",
      chunk_id: "lancelot-07-chunk-001",
      quote: "La Dame du Lac donna à Lancelot une bonne nourrice.",
      chapter_number: 7,
      chapter_title: "La Dame du Lac et Lancelot",
      source_url: SOURCE_URL,
    },
  ],
  retrieved_passages: passages(),
};

const insufficientResponse: AskResponse = {
  status: "insufficient",
  answer: "Les passages retrouvés ne permettent pas d’établir cette relation.",
  citations: [],
  retrieved_passages: passages(),
};

function mockResponse(payload: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(payload),
  } as unknown as Response;
}

function enterQuestion(question = "Qui recueille et élève Lancelot ?") {
  fireEvent.change(screen.getByLabelText("Votre question"), {
    target: { value: question },
  });
  return question;
}

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Les Archives arthuriennes", () => {
  it("presents the corpus and an empty initial consultation", () => {
    render(<App />);

    expect(
      screen.getByRole("heading", { name: "Les Archives arthuriennes" })
    ).toBeInTheDocument();
    expect(screen.getAllByText(/Les Enfances de Lancelot/)).toHaveLength(2);
    expect(screen.getByText(/dix-neuf chapitres/)).toBeInTheDocument();
    expect(screen.queryByText("Ce que disent les archives")).not.toBeInTheDocument();
  });

  it("does not submit a blank or whitespace-only question", () => {
    render(<App />);
    const textarea = screen.getByLabelText("Votre question");
    const submit = screen.getByRole("button", { name: "Consulter les archives" });

    expect(submit).toBeDisabled();
    fireEvent.change(textarea, { target: { value: "   " } });
    fireEvent.blur(textarea);
    fireEvent.keyDown(textarea, { key: "Enter" });

    expect(submit).toBeDisabled();
    expect(screen.getByText(/seulement des espaces/)).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("renders an answered response with fixed citation metadata", async () => {
    fetchMock.mockResolvedValue(mockResponse(answeredResponse));
    render(<App />);
    const question = enterQuestion();

    fireEvent.click(screen.getByRole("button", { name: "Consulter les archives" }));

    expect(await screen.findByText(answeredResponse.answer)).toBeInTheDocument();
    expect(screen.getByText("Question consultée").parentElement).toHaveTextContent(
      question
    );
    expect(screen.getByText("Citation 1")).toBeInTheDocument();
    expect(screen.getByText("evidence-01")).toBeInTheDocument();
    expect(screen.getAllByText("lancelot-07-chunk-001")).toHaveLength(2);
    expect(screen.getByText(answeredResponse.citations[0].quote)).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /Lire le chapitre sur Wikisource/ })
    ).toHaveAttribute("href", SOURCE_URL);
    expect(fetchMock).toHaveBeenCalledWith("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  });

  it("renders the cautious insufficient-evidence state without citations", async () => {
    fetchMock.mockResolvedValue(mockResponse(insufficientResponse));
    render(<App />);
    enterQuestion("Quelle est la relation entre Arthur et Lancelot ?");

    fireEvent.click(screen.getByRole("button", { name: "Consulter les archives" }));

    expect(await screen.findByText(insufficientResponse.answer)).toBeInTheDocument();
    expect(screen.getByText("Sources insuffisantes")).toBeInTheDocument();
    expect(screen.getByText(/volontairement prudente/)).toBeInTheDocument();
    expect(screen.queryByText("Citations reliées aux sources")).not.toBeInTheDocument();
  });

  it("keeps retrieval details collapsed and exposes all five passage records", async () => {
    fetchMock.mockResolvedValue(mockResponse(answeredResponse));
    render(<App />);
    enterQuestion();
    fireEvent.click(screen.getByRole("button", { name: "Consulter les archives" }));

    const summary = await screen.findByText("Comment cette réponse a été construite");
    const details = summary.closest("details");
    expect(details).not.toHaveAttribute("open");
    expect(screen.getByText("5 passages")).toBeInTheDocument();
    expect(screen.getByText("Passage retrouvé numéro 5.")).toBeInTheDocument();
    expect(screen.getByText("Similarité 0.8912")).toBeInTheDocument();
  });

  it("shows restrained loading feedback and prevents duplicate submissions", async () => {
    let resolveRequest: (response: Response) => void = () => undefined;
    fetchMock.mockReturnValue(
      new Promise<Response>((resolve) => {
        resolveRequest = resolve;
      })
    );
    render(<App />);
    enterQuestion();
    const form = screen.getByRole("form", { name: "Interroger les archives" });

    fireEvent.submit(form);
    expect(screen.getByText("Consultation des archives en cours")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Consultation en cours…" })).toBeDisabled();
    fireEvent.submit(form);
    expect(fetchMock).toHaveBeenCalledTimes(1);

    await act(async () => resolveRequest(mockResponse(answeredResponse)));
    expect(await screen.findByText(answeredResponse.answer)).toBeInTheDocument();
  });

  it("shows safe messages for backend and malformed-response failures", async () => {
    fetchMock.mockResolvedValueOnce(
      mockResponse({ detail: "Gemini configuration is unavailable." }, 503)
    );
    const { unmount } = render(<App />);
    enterQuestion();
    fireEvent.click(screen.getByRole("button", { name: "Consulter les archives" }));

    expect(
      await screen.findByText(/service local n’est pas prêt/)
    ).toBeInTheDocument();
    unmount();

    fetchMock.mockResolvedValueOnce(mockResponse({ status: "answered" }));
    render(<App />);
    enterQuestion();
    fireEvent.click(screen.getByRole("button", { name: "Consulter les archives" }));

    await waitFor(() =>
      expect(screen.getByText(/réponse inattendue/)).toBeInTheDocument()
    );
  });
});
