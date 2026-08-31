import type { AskResponse } from "../types";
import { CitationCard } from "./CitationCard";
import { RetrievalDetails } from "./RetrievalDetails";

interface AnswerCardProps {
  question: string;
  result: AskResponse;
}

export function AnswerCard({ question, result }: AnswerCardProps) {
  const answered = result.status === "answered";

  return (
    <article className={`answer-card ${answered ? "is-answered" : "is-insufficient"}`}>
      <header className="answer-header">
        <p className="eyebrow">
          {answered ? "Réponse établie" : "Sources insuffisantes"}
        </p>
        <h2>{answered ? "Ce que disent les archives" : "Ce que le corpus permet d’affirmer"}</h2>
        <p className="submitted-question">
          <span>Question consultée</span>
          {question}
        </p>
      </header>

      <div className="answer-copy">
        <p>{result.answer}</p>
      </div>

      {!answered && (
        <aside className="insufficient-note">
          Cette réponse reste volontairement prudente : les passages retrouvés ne
          permettent pas d’établir directement le fait demandé. Reformulez la
          question ou consultez les passages ci-dessous.
        </aside>
      )}

      {result.citations.length > 0 && (
        <section className="citations" aria-labelledby="citations-title">
          <div className="section-heading">
            <p className="eyebrow">Preuves textuelles</p>
            <h3 id="citations-title">Citations reliées aux sources</h3>
          </div>
          <div className="citation-grid">
            {result.citations.map((citation, index) => (
              <CitationCard
                key={citation.evidence_id}
                citation={citation}
                number={index + 1}
              />
            ))}
          </div>
        </section>
      )}

      <RetrievalDetails passages={result.retrieved_passages} />
    </article>
  );
}
