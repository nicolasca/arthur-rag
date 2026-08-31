import type { Citation } from "../types";

interface CitationCardProps {
  citation: Citation;
  number: number;
}

export function CitationCard({ citation, number }: CitationCardProps) {
  return (
    <details className="citation-card">
      <summary>
        <span className="citation-number">Citation {number}</span>
        <span className="citation-chapter">
          Chapitre {citation.chapter_number} · {citation.chapter_title}
        </span>
      </summary>
      <div className="citation-content">
        <blockquote>
          <p>{citation.quote}</p>
        </blockquote>
        <dl className="citation-metadata">
          <div>
            <dt>Preuve</dt>
            <dd>{citation.evidence_id}</dd>
          </div>
          <div>
            <dt>Fragment</dt>
            <dd>
              <code>{citation.chunk_id}</code>
            </dd>
          </div>
        </dl>
        <cite>
          <a href={citation.source_url} target="_blank" rel="noreferrer">
            Lire le chapitre sur Wikisource
            <span aria-hidden="true"> ↗</span>
          </a>
        </cite>
      </div>
    </details>
  );
}
