-- ch04 权威 DDL。存量库手动灌:docker exec -i mewhelp-mysql mysql --default-character-set=utf8mb4 -uroot -proot mewhelp < sql/ch04-ddl.sql
-- tests/conftest.py 按分号分割只跑 CREATE TABLE 句。
-- 注意:全文(含注释与字符串字面量)不得出现多余分号(COMMENT 文案内一律用全角分号),CREATE TABLE 句前不得有行内注释。
SET NAMES utf8mb4;

CREATE TABLE IF NOT EXISTS low_confidence_questions (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  conversation_id BIGINT NULL,
  raw_question TEXT NOT NULL,
  source ENUM('retrieval_low_conf','self_check','user_feedback') NOT NULL,
  reason TEXT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY idx_conversation (conversation_id),
  FOREIGN KEY (conversation_id) REFERENCES conversations(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='低置信问题池；本章写入 retrieval_low_conf 与 self_check 两来源,user_feedback 预留 ch09 飞轮。';

CREATE TABLE IF NOT EXISTS faith_cases (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  eval_id VARCHAR(64) NOT NULL COMMENT '一题一行',
  bucket VARCHAR(16) NOT NULL,
  query TEXT NOT NULL,
  strategy VARCHAR(32) NOT NULL,
  answer TEXT NOT NULL,
  reason TEXT NULL,
  citations JSON NULL COMMENT '这一轮 Top-K 证据全集快照,角标[n]即序号',
  judge_model VARCHAR(64) NULL,
  status ENUM('未解决','已解决','无需解决') NOT NULL DEFAULT '未解决',
  resolution TEXT NULL COMMENT '处置说明,标已解决/无需解决时必填——应用层校验',
  seen_count INT NOT NULL DEFAULT 1,
  first_seen_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_seen_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  resolved_at DATETIME NULL,
  UNIQUE KEY uk_eval_id (eval_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='编造个案跨轮台账；报告重跑即覆盖,个案与处置状态跨轮留存；复发退回未解决。';
