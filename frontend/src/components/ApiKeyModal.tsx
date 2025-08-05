import React, { useState } from 'react';
import { ApiKeys } from '../types';
import './ApiKeyModal.css';

interface ApiKeyModalProps {
  isOpen: boolean;
  onSubmit: (apiKeys: ApiKeys) => void;
}

const ApiKeyModal: React.FC<ApiKeyModalProps> = ({ isOpen, onSubmit }) => {
  const [openaiKey, setOpenaiKey] = useState('');
  const [tavilyKey, setTavilyKey] = useState('');
  const [cohereKey, setCohereKey] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!openaiKey.trim() || !tavilyKey.trim()) {
      alert('OpenAI and Tavily API keys are required!');
      return;
    }

    setIsLoading(true);
    
    try {
      await onSubmit({
        openai: openaiKey.trim(),
        tavily: tavilyKey.trim(),
        cohere: cohereKey.trim() || undefined
      });
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal">
        <h2 className="modal-title">Welcome to Roni</h2>
        <p className="modal-description">
          Please enter your API keys to get started with your AI Finance Assistant.
        </p>
        
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label" htmlFor="openaiKey">
              OpenAI API Key
            </label>
            <input
              type="password"
              id="openaiKey"
              className="form-input"
              placeholder="sk-..."
              value={openaiKey}
              onChange={(e) => setOpenaiKey(e.target.value)}
              required
            />
          </div>
          
          <div className="form-group">
            <label className="form-label" htmlFor="tavilyKey">
              Tavily API Key
            </label>
            <input
              type="password"
              id="tavilyKey"
              className="form-input"
              placeholder="tvly-..."
              value={tavilyKey}
              onChange={(e) => setTavilyKey(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="cohereKey">
              Cohere API Key (Optional)
            </label>
            <input
              type="password"
              id="cohereKey"
              className="form-input"
              placeholder="co-..."
              value={cohereKey}
              onChange={(e) => setCohereKey(e.target.value)}
            />
          </div>
          
          <div className="modal-actions">
            <button 
              type="submit" 
              className="btn-primary"
              disabled={isLoading}
            >
              {isLoading ? 'Starting...' : 'Start Chatting'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ApiKeyModal;