import React from 'react';
import './SuggestionButtons.css';

interface SuggestionButtonsProps {
  onSuggestionClick: (message: string) => void;
}

const SuggestionButtons: React.FC<SuggestionButtonsProps> = ({ onSuggestionClick }) => {
  const suggestions = [
    'How did I perform in June 2025?',
    'What business expenses are tax deductible?',
    'How do I calculate profit margins?',
    'When do I treat expenses as an asset?'
  ];

  return (
    <div className="suggestions">
      {suggestions.map((suggestion, index) => (
        <button
          key={index}
          className="suggestion-btn"
          onClick={() => onSuggestionClick(suggestion)}
        >
          {suggestion}
        </button>
      ))}
    </div>
  );
};

export default SuggestionButtons;