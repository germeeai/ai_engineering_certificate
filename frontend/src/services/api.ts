import { ApiKeys, PredictRequest, PredictResponse } from '../types';

const API_BASE_URL = 'http://localhost:8000';

class ApiService {
  private apiKeys: ApiKeys | null = null;

  setApiKeys(apiKeys: ApiKeys) {
    this.apiKeys = apiKeys;
  }

  async healthCheck(): Promise<{ status: string; message: string }> {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }
    return response.json();
  }

  async sendMessage(question: string): Promise<PredictResponse> {
    if (!this.apiKeys) {
      throw new Error('API keys not configured');
    }

    const request: PredictRequest = { question };
    
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-OpenAI-Key': this.apiKeys.openai,
        'X-Tavily-Key': this.apiKeys.tavily,
        ...(this.apiKeys.cohere && { 'X-Cohere-Key': this.apiKeys.cohere })
      },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API request failed: ${response.status} - ${errorText}`);
    }

    return response.json();
  }
}

export const apiService = new ApiService();