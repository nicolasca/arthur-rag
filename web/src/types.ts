export type AnswerStatus = "answered" | "insufficient";

export interface Citation {
  evidence_id: string;
  chunk_id: string;
  quote: string;
  chapter_number: number;
  chapter_title: string;
  source_url: string;
}

export interface RetrievedPassage {
  rank: number;
  chunk_id: string;
  similarity: number;
  chapter_number: number;
  chapter_title: string;
  source_url: string;
  text: string;
}

export interface AskResponse {
  status: AnswerStatus;
  answer: string;
  citations: Citation[];
  retrieved_passages: RetrievedPassage[];
}
