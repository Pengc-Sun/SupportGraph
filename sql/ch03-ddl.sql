-- ch03 权威 DDL。tests/conftest.py 按分号分割只跑 CREATE TABLE 句。
-- 注意:全文(含注释与字符串字面量)不得出现多余分号,CREATE TABLE 句前不得有行内注释。
SET NAMES utf8mb4;

CREATE TABLE knowledge_chunks (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  category VARCHAR(255) NOT NULL,
  questions TEXT NOT NULL,
  answer TEXT NOT NULL,
  section_path VARCHAR(512) NULL,
  content_type VARCHAR(32) NULL,
  is_key_clause INT NOT NULL DEFAULT 0,
  prev_chunk_id BIGINT NULL,
  next_chunk_id BIGINT NULL,
  vector_id VARCHAR(64) NULL,
  vectorize_status ENUM('pending','done') NOT NULL DEFAULT 'pending',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  KEY idx_vectorize_status (vectorize_status),
  FOREIGN KEY (prev_chunk_id) REFERENCES knowledge_chunks(id),
  FOREIGN KEY (next_chunk_id) REFERENCES knowledge_chunks(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE qa_extraction_staging (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  batch_no VARCHAR(64) NOT NULL,
  source_ref VARCHAR(255) NULL,
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  status ENUM('extracted','kept','discarded') NOT NULL DEFAULT 'extracted',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
