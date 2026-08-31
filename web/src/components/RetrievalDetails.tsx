import type { RetrievedPassage } from "../types";

interface RetrievalDetailsProps {
  passages: RetrievedPassage[];
}

export function RetrievalDetails({ passages }: RetrievalDetailsProps) {
  return (
    <details className="retrieval-details">
      <summary>
        <span>Comment cette réponse a été construite</span>
        <span className="passage-count">{passages.length} passages</span>
      </summary>
      <div className="retrieval-content">
        <p className="retrieval-note">
          La similarité cosinus indique une proximité relative avec la question;
          ce n’est ni une probabilité ni une preuve que le passage répond.
        </p>
        <ol className="passage-list">
          {passages.map((passage) => (
            <li key={`${passage.rank}-${passage.chunk_id}`}>
              <article className="passage-card">
                <header>
                  <span className="passage-rank">Rang {passage.rank}</span>
                  <span>Similarité {passage.similarity.toFixed(4)}</span>
                </header>
                <h4>
                  Chapitre {passage.chapter_number} · {passage.chapter_title}
                </h4>
                <p>{passage.text}</p>
                <footer>
                  <code>{passage.chunk_id}</code>
                  <a href={passage.source_url} target="_blank" rel="noreferrer">
                    Source Wikisource<span aria-hidden="true"> ↗</span>
                  </a>
                </footer>
              </article>
            </li>
          ))}
        </ol>
      </div>
    </details>
  );
}
