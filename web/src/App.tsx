import { FormEvent, KeyboardEvent, useState } from "react";
import { askQuestion } from "./api";
import { AnswerCard } from "./components/AnswerCard";
import type { AskResponse } from "./types";

const MAX_QUESTION_LENGTH = 500;
const EXAMPLE_QUESTIONS = [
  { question: "Qui recueille et élève Lancelot ?" },
  { question: "Pourquoi le roi Ban quitte-t-il Trèbe ?" },
  { question: "Quelle relation unit Lionel et Bohor ?" },
  {
    question: "Lancelot entretient-il une relation amoureuse ?",
    cautious: true,
  },
];

export default function App() {
  const [question, setQuestion] = useState("");
  const [submittedQuestion, setSubmittedQuestion] = useState("");
  const [result, setResult] = useState<AskResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [touched, setTouched] = useState(false);

  const trimmedQuestion = question.trim();
  const remaining = MAX_QUESTION_LENGTH - question.length;
  const blankAfterTyping = touched && question.length > 0 && !trimmedQuestion;
  const canSubmit = Boolean(trimmedQuestion) && !isLoading;

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setTouched(true);
    if (!trimmedQuestion || isLoading) return;

    setIsLoading(true);
    setError("");
    setResult(null);
    setSubmittedQuestion(trimmedQuestion);

    try {
      const response = await askQuestion(trimmedQuestion);
      setResult(response);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Une erreur inattendue a interrompu la consultation."
      );
    } finally {
      setIsLoading(false);
    }
  }

  function handleQuestionKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) {
      return;
    }
    event.preventDefault();
    if (canSubmit) event.currentTarget.form?.requestSubmit();
  }

  function chooseExample(example: string) {
    setQuestion(example);
    setTouched(false);
    setError("");
  }

  return (
    <div className="site-shell">
      <a className="skip-link" href="#consultation">
        Aller à la consultation
      </a>

      <header className="masthead">
        <div className="archive-mark" aria-hidden="true">
          <span>AA</span>
        </div>
        <div>
          <p className="collection-label">Bibliothèque expérimentale · Corpus I</p>
          <h1>Les Archives arthuriennes</h1>
          <p className="subtitle">
            Corpus actuel : <cite>Les Enfances de Lancelot</cite>, adaptation de
            Jacques Boulenger (1922)
          </p>
        </div>
      </header>

      <main id="consultation">
        <section className="question-panel" aria-labelledby="question-title">
          <div className="question-intro">
            <p className="eyebrow">Consultation du manuscrit</p>
            <h2 id="question-title">Que souhaitez-vous chercher ?</h2>
            <p>
              Posez une question précise. La réponse sera limitée aux dix-neuf
              chapitres du corpus et accompagnée de ses preuves textuelles.
            </p>
          </div>

          <form aria-label="Interroger les archives" onSubmit={handleSubmit}>
            <label htmlFor="archive-question">Votre question</label>
            <textarea
              id="archive-question"
              name="question"
              rows={4}
              maxLength={MAX_QUESTION_LENGTH}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onBlur={() => setTouched(true)}
              onKeyDown={handleQuestionKeyDown}
              aria-describedby="question-guidance question-counter question-validation"
              aria-invalid={blankAfterTyping}
              placeholder="Ex. Qui recueille et élève Lancelot ?"
            />
            <div className="field-footer">
              <span id="question-guidance">Entrée pour envoyer · Maj + Entrée pour une nouvelle ligne</span>
              <span id="question-counter" className={remaining < 50 ? "counter-warning" : ""}>
                {remaining} caractères restants
              </span>
            </div>
            <p id="question-validation" className="validation-message" aria-live="polite">
              {blankAfterTyping ? "La question ne peut pas contenir seulement des espaces." : ""}
            </p>
            <button className="primary-action" type="submit" disabled={!canSubmit}>
              {isLoading ? "Consultation en cours…" : "Consulter les archives"}
            </button>
          </form>

          <div className="examples" aria-labelledby="examples-title">
            <h3 id="examples-title">Exemples de questions</h3>
            <div className="example-list">
              {EXAMPLE_QUESTIONS.map((example) => (
                <button
                  key={example.question}
                  type="button"
                  className="example-button"
                  onClick={() => chooseExample(example.question)}
                  disabled={isLoading}
                >
                  <span>{example.question}</span>
                  {example.cautious && <small>Peut être insuffisant</small>}
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className="result-region" aria-live="polite" aria-busy={isLoading}>
          {isLoading && (
            <div className="loading-card" role="status">
              <span className="loading-ornament" aria-hidden="true">✦</span>
              <div>
                <h2>Consultation des archives en cours</h2>
                <p>Recherche de cinq passages, puis rédaction d’une réponse ancrée…</p>
                <p className="loading-question">{submittedQuestion}</p>
              </div>
            </div>
          )}

          {error && (
            <div className="error-card" role="alert">
              <p className="eyebrow">Consultation interrompue</p>
              <h2>Les archives ne sont pas disponibles</h2>
              <p>{error}</p>
              {submittedQuestion && (
                <p className="submitted-question compact">
                  <span>Question tentée</span>
                  {submittedQuestion}
                </p>
              )}
            </div>
          )}

          {result && <AnswerCard question={submittedQuestion} result={result} />}
        </section>
      </main>

      <footer className="site-footer">
        <div>
          <h2>Limites du fonds</h2>
          <p>
            Ce prototype ne consulte que <cite>Les Enfances de Lancelot</cite>. Une
            absence de réponse ne vaut pas absence dans toute la tradition arthurienne.
          </p>
        </div>
        <div>
          <h2>Confidentialité</h2>
          <p>
            Les questions sont envoyées à Google Gemini par le serveur. N’envoyez
            aucune information privée ou confidentielle.
          </p>
        </div>
        <a
          className="wikisource-link"
          href="https://fr.wikisource.org/wiki/Les_Enfances_de_Lancelot"
          target="_blank"
          rel="noreferrer"
        >
          Consulter l’œuvre sur Wikisource<span aria-hidden="true"> ↗</span>
        </a>
      </footer>
    </div>
  );
}
