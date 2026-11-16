-- ch02 权威 DDL。docker initdb 首启执行,tests/conftest.py 按分号分割只跑 CREATE TABLE 句。
-- 注意:全文(含注释与字符串字面量)不得出现多余分号,CREATE TABLE 句前不得有行内注释。
SET NAMES utf8mb4;

CREATE TABLE conversations (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id VARCHAR(64) NOT NULL,
  status ENUM('进行中','已转人工','已结束') NOT NULL DEFAULT '进行中',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE messages (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  conversation_id BIGINT NOT NULL,
  role ENUM('user','assistant','tool') NOT NULL,
  content TEXT NULL,
  tool_calls JSON NULL,
  tool_call_id VARCHAR(64) NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY idx_conversation (conversation_id),
  FOREIGN KEY (conversation_id) REFERENCES conversations(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE faq (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  question VARCHAR(512) NOT NULL,
  answer TEXT NOT NULL,
  category VARCHAR(64) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE tickets (
  ticket_no VARCHAR(32) PRIMARY KEY,
  conversation_id BIGINT NOT NULL,
  description TEXT NOT NULL,
  ticket_type ENUM('售后','投诉','咨询') NOT NULL,
  status ENUM('待处理','已处理') NOT NULL DEFAULT '待处理',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (conversation_id) REFERENCES conversations(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
