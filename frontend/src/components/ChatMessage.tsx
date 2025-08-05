import React from 'react';
import { Message } from '../types';
import './ChatMessage.css';

interface ChatMessageProps {
  message: Message;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  return (
    <div className={`message ${message.isUser ? 'user-message' : ''}`}>
      <div className="message-avatar">
        {message.isUser ? '👤' : '🤖'}
      </div>
      <div className="message-content">
        <div className="message-text">
          {message.isLoading ? (
            <div className="loading-spinner"></div>
          ) : (
            message.content
          )}
        </div>
        <div className="message-time">
          {formatTime(message.timestamp)}
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;