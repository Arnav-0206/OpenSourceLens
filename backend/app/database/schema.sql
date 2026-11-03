-- ========================================================
-- OpenSourceLens Database Schema (POSTGRESQL)
-- Auto-generated from SQLAlchemy 2.0 declarative models
-- ========================================================

CREATE TABLE repositories (
	id BIGINT NOT NULL, 
	repo_name VARCHAR(120) NOT NULL, 
	owner VARCHAR(80) NOT NULL, 
	name VARCHAR(80) NOT NULL, 
	description TEXT, 
	primary_language VARCHAR(60), 
	stars_count INTEGER NOT NULL, 
	forks_count INTEGER NOT NULL, 
	open_issues_count INTEGER NOT NULL, 
	github_created_at TIMESTAMP WITH TIME ZONE, 
	github_updated_at TIMESTAMP WITH TIME ZONE, 
	collected_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	CONSTRAINT pk_repositories PRIMARY KEY (id)
);

CREATE UNIQUE INDEX ix_repositories_repo_name ON repositories (repo_name);

CREATE TABLE issues (
	id BIGINT NOT NULL, 
	repository_id BIGINT NOT NULL, 
	repo_name VARCHAR(120) NOT NULL, 
	issue_number INTEGER NOT NULL, 
	title TEXT NOT NULL, 
	body TEXT, 
	clean_title TEXT, 
	clean_body TEXT, 
	combined_text TEXT, 
	clean_title_word_count INTEGER NOT NULL, 
	clean_body_word_count INTEGER NOT NULL, 
	state VARCHAR(30) NOT NULL, 
	state_reason VARCHAR(50), 
	author VARCHAR(100) NOT NULL, 
	author_type VARCHAR(40) NOT NULL, 
	author_association VARCHAR(40) NOT NULL, 
	is_internal_author BOOLEAN NOT NULL, 
	is_bot_author BOOLEAN NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE, 
	closed_at TIMESTAMP WITH TIME ZONE, 
	comments_count INTEGER NOT NULL, 
	reactions_count INTEGER NOT NULL, 
	labels_count INTEGER NOT NULL, 
	has_code_block BOOLEAN NOT NULL, 
	code_block_count INTEGER NOT NULL, 
	has_url BOOLEAN NOT NULL, 
	has_stack_trace BOOLEAN NOT NULL, 
	canonical_label VARCHAR(50) NOT NULL, 
	canonical_label_id INTEGER NOT NULL, 
	is_classified_category BOOLEAN NOT NULL, 
	is_unlabeled BOOLEAN NOT NULL, 
	resolution_time_hours FLOAT, 
	resolution_time_days FLOAT, 
	speed_tier VARCHAR(30), 
	resolution_type VARCHAR(30), 
	is_censored BOOLEAN NOT NULL, 
	is_immediate_closure BOOLEAN NOT NULL, 
	duration_at_risk_days FLOAT, 
	CONSTRAINT pk_issues PRIMARY KEY (id), 
	CONSTRAINT fk_issues_repository_id_repositories FOREIGN KEY(repository_id) REFERENCES repositories (id) ON DELETE CASCADE
);

CREATE INDEX ix_issues_canonical_label ON issues (canonical_label);

CREATE INDEX ix_issues_created_at ON issues (created_at);

CREATE INDEX ix_issues_speed_tier ON issues (speed_tier);

CREATE INDEX ix_issues_repo_canonical ON issues (repo_name, canonical_label);

CREATE INDEX ix_issues_repo_state ON issues (repo_name, state);

CREATE INDEX ix_issues_repository_id ON issues (repository_id);

CREATE INDEX ix_issues_closed_at ON issues (closed_at);

CREATE INDEX ix_issues_state ON issues (state);

CREATE INDEX ix_issues_repo_speed_tier ON issues (repo_name, speed_tier);

CREATE UNIQUE INDEX ix_issues_repo_number ON issues (repo_name, issue_number);

CREATE INDEX ix_issues_repo_name ON issues (repo_name);

CREATE TABLE repository_analytics (
	id BIGSERIAL NOT NULL, 
	repository_id BIGINT NOT NULL, 
	repo_name VARCHAR(120) NOT NULL, 
	snapshot_date TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	total_issues INTEGER NOT NULL, 
	open_issues INTEGER NOT NULL, 
	closed_issues INTEGER NOT NULL, 
	category_distribution JSON, 
	median_resolution_hours FLOAT, 
	mean_resolution_hours FLOAT, 
	p25_resolution_hours FLOAT, 
	p75_resolution_hours FLOAT, 
	survival_rate_30d FLOAT, 
	speed_tier_distribution JSON, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	CONSTRAINT pk_repository_analytics PRIMARY KEY (id), 
	CONSTRAINT fk_repository_analytics_repository_id_repositories FOREIGN KEY(repository_id) REFERENCES repositories (id) ON DELETE CASCADE
);

CREATE INDEX ix_analytics_repo_date ON repository_analytics (repository_id, snapshot_date);

CREATE INDEX ix_repository_analytics_repository_id ON repository_analytics (repository_id);

CREATE INDEX ix_repository_analytics_repo_name ON repository_analytics (repo_name);

CREATE TABLE issue_labels (
	id BIGSERIAL NOT NULL, 
	issue_id BIGINT NOT NULL, 
	label_name VARCHAR(100) NOT NULL, 
	is_canonical BOOLEAN NOT NULL, 
	canonical_category VARCHAR(50), 
	CONSTRAINT pk_issue_labels PRIMARY KEY (id), 
	CONSTRAINT fk_issue_labels_issue_id_issues FOREIGN KEY(issue_id) REFERENCES issues (id) ON DELETE CASCADE
);

CREATE INDEX ix_issue_labels_issue_id ON issue_labels (issue_id);

CREATE INDEX ix_issue_labels_label_name ON issue_labels (label_name);

CREATE TABLE predictions (
	id BIGSERIAL NOT NULL, 
	issue_id BIGINT, 
	repo_name VARCHAR(120), 
	input_title TEXT NOT NULL, 
	input_body TEXT, 
	model_name VARCHAR(80) NOT NULL, 
	task_type VARCHAR(50) NOT NULL, 
	predicted_label VARCHAR(80), 
	confidence_score FLOAT, 
	probabilities JSON, 
	inference_time_ms FLOAT NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	CONSTRAINT pk_predictions PRIMARY KEY (id), 
	CONSTRAINT fk_predictions_issue_id_issues FOREIGN KEY(issue_id) REFERENCES issues (id) ON DELETE SET NULL
);

CREATE INDEX ix_predictions_issue_id ON predictions (issue_id);

CREATE TABLE related_issues_cache (
	id BIGSERIAL NOT NULL, 
	source_issue_id BIGINT NOT NULL, 
	target_issue_id BIGINT NOT NULL, 
	similarity_score FLOAT NOT NULL, 
	rank INTEGER NOT NULL, 
	model_version VARCHAR(60) NOT NULL, 
	calculated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	CONSTRAINT pk_related_issues_cache PRIMARY KEY (id), 
	CONSTRAINT uq_related_source_target UNIQUE (source_issue_id, target_issue_id), 
	CONSTRAINT fk_related_issues_cache_source_issue_id_issues FOREIGN KEY(source_issue_id) REFERENCES issues (id) ON DELETE CASCADE, 
	CONSTRAINT fk_related_issues_cache_target_issue_id_issues FOREIGN KEY(target_issue_id) REFERENCES issues (id) ON DELETE CASCADE
);

CREATE INDEX ix_related_issues_cache_target_issue_id ON related_issues_cache (target_issue_id);

CREATE INDEX ix_related_issues_cache_source_issue_id ON related_issues_cache (source_issue_id);

CREATE INDEX ix_related_source_rank ON related_issues_cache (source_issue_id, rank);
