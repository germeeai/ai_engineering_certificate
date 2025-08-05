import React, { useState, useEffect, useRef } from 'react';
import { Message, ApiKeys } from './types';
import { apiService } from './services/api';
import ApiKeyModal from './components/ApiKeyModal';
import ChatMessage from './components/ChatMessage';
import SuggestionButtons from './components/SuggestionButtons';
import ChatInput from './components/ChatInput';
import './App.css';

const App: React.FC = () => {
  const [isApiModalOpen, setIsApiModalOpen] = useState(true);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Initialize with welcome messages
    const welcomeMessages: Message[] = [
      {
        id: '1',
        content: "Hello! I'm Roni, your AI Finance Assistant. How can I help you?",
        isUser: false,
        timestamp: new Date()
      },
      {
        id: '2',
        content: "Here are some questions you can ask me:",
        isUser: false,
        timestamp: new Date()
      }
    ];
    setMessages(welcomeMessages);
  }, []);

  const handleApiKeysSubmit = async (apiKeys: ApiKeys) => {
    try {
      apiService.setApiKeys(apiKeys);
      
      // Test the connection
      await apiService.healthCheck();
      
      setIsApiModalOpen(false);
    } catch (error) {
      console.error('Failed to initialize API:', error);
      alert('Failed to connect to the backend. Please make sure the server is running on localhost:8000');
    }
  };

  const handleSendMessage = async (messageContent: string) => {
    if (isLoading) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      content: messageContent,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setShowSuggestions(false);
    setIsLoading(true);

    // Add loading message
    const loadingMessage: Message = {
      id: (Date.now() + 1).toString(),
      content: '',
      isUser: false,
      timestamp: new Date(),
      isLoading: true
    };

    setMessages(prev => [...prev, loadingMessage]);

    try {
      const response = await apiService.sendMessage(messageContent);
      
      // Remove loading message and add actual response
      setMessages(prev => {
        const filtered = prev.filter(msg => !msg.isLoading);
        const botMessage: Message = {
          id: (Date.now() + 2).toString(),
          content: response.response || "I don't know.",
          isUser: false,
          timestamp: new Date()
        };
        return [...filtered, botMessage];
      });

    } catch (error) {
      console.error('Error sending message:', error);
      
      // Remove loading message and add error message
      setMessages(prev => {
        const filtered = prev.filter(msg => !msg.isLoading);
        const errorMessage: Message = {
          id: (Date.now() + 2).toString(),
          content: "Sorry, I'm having trouble connecting to the server. Please make sure the backend is running and try again.",
          isUser: false,
          timestamp: new Date()
        };
        return [...filtered, errorMessage];
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    handleSendMessage(suggestion);
  };

  const handleUpdateApiKeys = () => {
    setIsApiModalOpen(true);
  };

  return (
    <div className="app">
      <ApiKeyModal 
        isOpen={isApiModalOpen} 
        onSubmit={handleApiKeysSubmit} 
      />

      <header className="header">
        <div>
          <div className="breadcrumb">
            <span className="home-icon">🏠</span>
            <span className="separator">›</span>
            <span>Chat</span>
          </div>
          <h1 className="title">Chat with Roni</h1>
        </div>
        <button className="upload-btn" onClick={handleUpdateApiKeys}>
          <span>⚙️</span>
          Update API Keys
        </button>
      </header>

      <div className="chat-container">
        <div className="messages">
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
          
          {showSuggestions && (
            <div className="message">
              <div className="message-avatar">🤖</div>
              <div className="message-content">
                <SuggestionButtons onSuggestionClick={handleSuggestionClick} />
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      <ChatInput 
        onSendMessage={handleSendMessage} 
        disabled={isLoading || isApiModalOpen}
      />
    </div>
  );
};

export default App;