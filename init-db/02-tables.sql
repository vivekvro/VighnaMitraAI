CREATE TABLE IF NOT EXISTS accounts_info (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE,
    password BYTEA,
    dob DATE,
    email TEXT UNIQUE
);
CREATE TABLE uploaded_documents (
    id BIGSERIAL PRIMARY KEY,

    user_id VARCHAR(255) NOT NULL,
    thread_id VARCHAR(255) NOT NULL,

    file_name TEXT NOT NULL,
    file_hash CHAR(64) NOT NULL,

    summary TEXT,

    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_thread_file_hash
        UNIQUE (thread_id, file_hash)
);