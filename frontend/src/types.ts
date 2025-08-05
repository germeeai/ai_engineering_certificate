export interface ApiKeys {
  openai: string;
  tavily: string;
  cohere?: string;
}

export interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: Date;
  isLoading?: boolean;
}

export interface PredictRequest {
  question: string;
}

export interface PredictResponse {
  response: string;
  context?: string[];
}