-- Messages table
CREATE TABLE chat_messages (
    id SERIAL PRIMARY KEY,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    model_type VARCHAR(50) NULL,
    tokens_used INTEGER DEFAULT 0,
    response_time_ms INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Feedback table
CREATE TABLE message_feedback (
    id SERIAL PRIMARY KEY,
    message_id INTEGER REFERENCES chat_messages(id),
    rating BOOLEAN NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS message_contexts (
    id SERIAL PRIMARY KEY,
    message_id INTEGER REFERENCES chat_messages(id),
    context_text TEXT NOT NULL,
    relevance_score FLOAT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_chat_messages_timestamp ON chat_messages(timestamp);
CREATE INDEX idx_message_feedback_message_id ON message_feedback(message_id);
CREATE INDEX idx_message_contexts_message_id ON message_contexts(message_id);
