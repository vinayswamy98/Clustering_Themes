# 📊 ExamForge AI - Database Schema

## Overview

PostgreSQL database schema designed for Supabase with Row-Level Security (RLS) enabled.

---

## 🗂️ Schema Diagram

```
┌──────────────────┐       ┌──────────────────┐
│      users       │       │    exam_types    │
├──────────────────┤       ├──────────────────┤
│ id (PK)          │       │ id (PK)          │
│ email            │       │ name             │
│ name             │       │ description      │
│ exam_type_id (FK)│──────▶│ total_marks      │
│ target_date      │       │ duration_minutes │
│ onboarding_done  │       │ sections (JSONB) │
│ created_at       │       └──────────────────┘
│ updated_at       │
└────────┬─────────┘
         │
         │ 1:N
         ▼
┌──────────────────┐       ┌──────────────────┐
│  user_profiles   │       │    questions     │
├──────────────────┤       ├──────────────────┤
│ id (PK)          │       │ id (PK)          │
│ user_id (FK)     │       │ exam_type_id (FK)│
│ avatar_url       │       │ subject          │
│ streak_count     │       │ topic            │
│ max_streak       │       │ subtopic         │
│ total_xp         │       │ difficulty       │
│ current_level    │       │ elo_rating       │
│ timezone         │       │ question_type    │
│ daily_goal_mins  │       │ question_text    │
│ preferred_study  │       │ question_image   │
│ notifications    │       │ options (JSONB)  │
└──────────────────┘       │ correct_answer   │
         │                 │ solution_text    │
         │ 1:N             │ solution_image   │
         ▼                 │ hints (JSONB)    │
┌──────────────────┐       │ tags (ARRAY)     │
│    attempts      │       │ year             │
├──────────────────┤       │ is_verified      │
│ id (PK)          │       │ created_at       │
│ user_id (FK)     │──────▶└──────────────────┘
│ question_id (FK) │
│ session_id (FK)  │
│ selected_answer  │
│ is_correct       │
│ time_taken_secs  │
│ confidence_level │
│ hint_used        │
│ created_at       │
└──────────────────┘
         │
         │ N:1
         ▼
┌──────────────────┐       ┌──────────────────┐
│  study_sessions  │       │  mastery_states  │
├──────────────────┤       ├──────────────────┤
│ id (PK)          │       │ id (PK)          │
│ user_id (FK)     │       │ user_id (FK)     │
│ session_type     │       │ topic            │
│ started_at       │       │ subtopic         │
│ ended_at         │       │ elo_rating       │
│ total_questions  │       │ confidence       │
│ correct_answers  │       │ total_attempts   │
│ xp_earned        │       │ correct_attempts │
│ topics_covered   │       │ last_practiced   │
│ metadata (JSONB) │       │ next_review_at   │
└──────────────────┘       │ streak_count     │
                           │ created_at       │
                           │ updated_at       │
                           └──────────────────┘
```

---

## 📋 Table Definitions

### 1. `exam_types`

Defines supported exam types with their configurations.

```sql
CREATE TABLE exam_types (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    display_name VARCHAR(200) NOT NULL,
    description TEXT,
    country VARCHAR(100),
    total_marks INTEGER,
    duration_minutes INTEGER,
    negative_marking DECIMAL(3,2) DEFAULT 0, -- e.g., 0.25 for JEE
    sections JSONB, -- Subject-wise breakdown
    passing_criteria JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Example: JEE Main
INSERT INTO exam_types (name, display_name, description, country, total_marks, duration_minutes, negative_marking, sections) VALUES
('jee_main', 'JEE Main 2026', 'Joint Entrance Examination Main', 'India', 300, 180, 0.25,
 '{"physics": {"questions": 30, "marks": 100}, "chemistry": {"questions": 30, "marks": 100}, "mathematics": {"questions": 30, "marks": 100}}'
);
```

### 2. `users`

Core user table with authentication linkage.

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255),
    exam_type_id UUID REFERENCES exam_types(id),
    target_date DATE, -- Target exam date
    target_score INTEGER, -- Target percentile/score
    current_estimated_score INTEGER,
    onboarding_completed BOOLEAN DEFAULT false,
    diagnostic_completed BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- RLS Policy
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own data" ON users
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own data" ON users
    FOR UPDATE USING (auth.uid() = id);
```

### 3. `user_profiles`

Extended user profile with gamification data.

```sql
CREATE TABLE user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    avatar_url TEXT,
    
    -- Gamification
    streak_count INTEGER DEFAULT 0,
    max_streak INTEGER DEFAULT 0,
    last_activity_date DATE,
    total_xp INTEGER DEFAULT 0,
    current_level INTEGER DEFAULT 1,
    
    -- Preferences
    timezone VARCHAR(100) DEFAULT 'Asia/Kolkata',
    daily_goal_minutes INTEGER DEFAULT 30,
    preferred_study_time VARCHAR(50), -- 'morning', 'afternoon', 'evening', 'night'
    notification_enabled BOOLEAN DEFAULT true,
    notification_time TIME DEFAULT '08:00:00',
    
    -- Statistics
    total_questions_attempted INTEGER DEFAULT 0,
    total_correct_answers INTEGER DEFAULT 0,
    total_study_minutes INTEGER DEFAULT 0,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(user_id)
);

-- RLS Policy
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile" ON user_profiles
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can update own profile" ON user_profiles
    FOR UPDATE USING (auth.uid() = user_id);
```

### 4. `topics`

Hierarchical topic structure for each exam.

```sql
CREATE TABLE topics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exam_type_id UUID NOT NULL REFERENCES exam_types(id),
    subject VARCHAR(100) NOT NULL, -- Physics, Chemistry, Math
    topic_name VARCHAR(255) NOT NULL,
    subtopic_name VARCHAR(255),
    parent_topic_id UUID REFERENCES topics(id),
    difficulty_weight DECIMAL(3,2) DEFAULT 1.0, -- Weight in exam
    prerequisite_topics UUID[], -- Array of prerequisite topic IDs
    description TEXT,
    order_index INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_topics_exam_subject ON topics(exam_type_id, subject);
```

### 5. `questions`

Question bank with comprehensive metadata.

```sql
CREATE TABLE questions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exam_type_id UUID NOT NULL REFERENCES exam_types(id),
    topic_id UUID REFERENCES topics(id),
    
    -- Classification
    subject VARCHAR(100) NOT NULL,
    topic VARCHAR(255) NOT NULL,
    subtopic VARCHAR(255),
    tags TEXT[] DEFAULT '{}',
    
    -- Difficulty
    difficulty INTEGER NOT NULL CHECK (difficulty BETWEEN 1 AND 5),
    elo_rating INTEGER DEFAULT 1200,
    
    -- Question content
    question_type VARCHAR(50) NOT NULL, -- 'single', 'multiple', 'numerical', 'matrix'
    question_text TEXT NOT NULL,
    question_image_url TEXT,
    question_latex TEXT, -- For complex math rendering
    
    -- Options (for MCQ)
    options JSONB, -- [{"id": "A", "text": "...", "image_url": null}]
    
    -- Answer
    correct_answer JSONB NOT NULL, -- {"answer": "A"} or {"answer": 25.5, "tolerance": 0.1}
    solution_text TEXT,
    solution_image_url TEXT,
    solution_video_url TEXT,
    
    -- Hints (progressive)
    hints JSONB, -- [{"level": 1, "text": "..."}, {"level": 2, "text": "..."}]
    
    -- Metadata
    source VARCHAR(255), -- 'JEE 2023', 'NCERT', 'Custom'
    year INTEGER,
    is_previous_year BOOLEAN DEFAULT false,
    is_verified BOOLEAN DEFAULT false,
    verified_by UUID,
    
    -- Statistics
    total_attempts INTEGER DEFAULT 0,
    correct_attempts INTEGER DEFAULT 0,
    avg_time_seconds INTEGER,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_questions_exam_topic ON questions(exam_type_id, topic);
CREATE INDEX idx_questions_difficulty ON questions(difficulty, elo_rating);
CREATE INDEX idx_questions_tags ON questions USING GIN(tags);
```

### 6. `study_sessions`

Track user study sessions.

```sql
CREATE TABLE study_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    session_type VARCHAR(50) NOT NULL, -- 'practice', 'mock_test', 'review', 'diagnostic'
    exam_type_id UUID REFERENCES exam_types(id),
    
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    
    -- Statistics
    total_questions INTEGER DEFAULT 0,
    correct_answers INTEGER DEFAULT 0,
    wrong_answers INTEGER DEFAULT 0,
    skipped_questions INTEGER DEFAULT 0,
    
    -- Results
    score INTEGER,
    max_score INTEGER,
    percentile DECIMAL(5,2),
    xp_earned INTEGER DEFAULT 0,
    
    -- Coverage
    topics_covered TEXT[] DEFAULT '{}',
    difficulty_distribution JSONB, -- {"easy": 10, "medium": 15, "hard": 5}
    
    -- Metadata
    metadata JSONB, -- Any additional session data
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_sessions_user_date ON study_sessions(user_id, created_at);
CREATE INDEX idx_sessions_type ON study_sessions(session_type);

-- RLS Policy
ALTER TABLE study_sessions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own sessions" ON study_sessions
    FOR SELECT USING (auth.uid() = user_id);
```

### 7. `attempts`

Individual question attempts.

```sql
CREATE TABLE attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    question_id UUID NOT NULL REFERENCES questions(id),
    session_id UUID REFERENCES study_sessions(id),
    
    -- Answer
    selected_answer JSONB, -- User's answer
    is_correct BOOLEAN,
    is_skipped BOOLEAN DEFAULT false,
    
    -- Timing
    time_taken_seconds INTEGER,
    started_at TIMESTAMPTZ,
    submitted_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Engagement
    hint_level_used INTEGER DEFAULT 0, -- 0 = no hint, 1+ = hint level
    solution_viewed BOOLEAN DEFAULT false,
    confidence_level INTEGER CHECK (confidence_level BETWEEN 1 AND 5),
    
    -- Mastery impact
    elo_change INTEGER,
    mastery_before DECIMAL(5,4),
    mastery_after DECIMAL(5,4),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_attempts_user_question ON attempts(user_id, question_id);
CREATE INDEX idx_attempts_user_date ON attempts(user_id, created_at);
CREATE INDEX idx_attempts_session ON attempts(session_id);

-- RLS Policy
ALTER TABLE attempts ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own attempts" ON attempts
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own attempts" ON attempts
    FOR INSERT WITH CHECK (auth.uid() = user_id);
```

### 8. `mastery_states`

User's mastery level per topic (Bayesian Knowledge Tracing).

```sql
CREATE TABLE mastery_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic_id UUID REFERENCES topics(id),
    
    -- Topic identification
    subject VARCHAR(100) NOT NULL,
    topic VARCHAR(255) NOT NULL,
    subtopic VARCHAR(255),
    
    -- Mastery metrics
    elo_rating INTEGER DEFAULT 1200,
    mastery_probability DECIMAL(5,4) DEFAULT 0.5, -- P(learned)
    confidence_interval DECIMAL(5,4) DEFAULT 0.3, -- Uncertainty
    
    -- Bayesian parameters
    p_init DECIMAL(5,4) DEFAULT 0.0, -- Initial knowledge
    p_transit DECIMAL(5,4) DEFAULT 0.1, -- Learning rate
    p_slip DECIMAL(5,4) DEFAULT 0.1, -- Known but wrong
    p_guess DECIMAL(5,4) DEFAULT 0.25, -- Unknown but right
    
    -- Statistics
    total_attempts INTEGER DEFAULT 0,
    correct_attempts INTEGER DEFAULT 0,
    recent_accuracy DECIMAL(5,4), -- Last 10 attempts
    
    -- Spaced repetition
    last_practiced_at TIMESTAMPTZ,
    next_review_at TIMESTAMPTZ,
    review_interval_days INTEGER DEFAULT 1,
    ease_factor DECIMAL(4,2) DEFAULT 2.5,
    
    -- Streak within topic
    current_streak INTEGER DEFAULT 0,
    max_streak INTEGER DEFAULT 0,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(user_id, topic, subtopic)
);

CREATE INDEX idx_mastery_user_topic ON mastery_states(user_id, topic);
CREATE INDEX idx_mastery_next_review ON mastery_states(next_review_at);

-- RLS Policy
ALTER TABLE mastery_states ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own mastery" ON mastery_states
    FOR SELECT USING (auth.uid() = user_id);
```

### 9. `achievements`

Achievement/badge definitions.

```sql
CREATE TABLE achievements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    icon_url TEXT,
    category VARCHAR(100), -- 'streak', 'mastery', 'practice', 'social'
    
    -- Requirements
    requirement_type VARCHAR(100), -- 'streak_days', 'total_questions', 'topic_mastery'
    requirement_value INTEGER,
    requirement_metadata JSONB,
    
    -- Rewards
    xp_reward INTEGER DEFAULT 0,
    
    is_hidden BOOLEAN DEFAULT false, -- Hidden until earned
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Example achievements
INSERT INTO achievements (code, name, description, category, requirement_type, requirement_value, xp_reward) VALUES
('first_question', 'First Step', 'Answer your first question', 'practice', 'total_questions', 1, 10),
('streak_7', 'Week Warrior', 'Maintain a 7-day streak', 'streak', 'streak_days', 7, 100),
('streak_30', 'Monthly Master', 'Maintain a 30-day streak', 'streak', 'streak_days', 30, 500),
('physics_master', 'Physics Prodigy', 'Reach 90% mastery in Physics', 'mastery', 'subject_mastery', 90, 1000);
```

### 10. `user_achievements`

User's earned achievements.

```sql
CREATE TABLE user_achievements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    achievement_id UUID NOT NULL REFERENCES achievements(id),
    earned_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(user_id, achievement_id)
);

CREATE INDEX idx_user_achievements_user ON user_achievements(user_id);

-- RLS Policy
ALTER TABLE user_achievements ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own achievements" ON user_achievements
    FOR SELECT USING (auth.uid() = user_id);
```

### 11. `daily_goals`

Daily activity tracking.

```sql
CREATE TABLE daily_goals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    date DATE NOT NULL DEFAULT CURRENT_DATE,
    
    -- Goals
    goal_minutes INTEGER DEFAULT 30,
    goal_questions INTEGER DEFAULT 20,
    goal_xp INTEGER DEFAULT 100,
    
    -- Progress
    actual_minutes INTEGER DEFAULT 0,
    actual_questions INTEGER DEFAULT 0,
    actual_xp INTEGER DEFAULT 0,
    
    -- Completion
    is_completed BOOLEAN DEFAULT false,
    completed_at TIMESTAMPTZ,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(user_id, date)
);

CREATE INDEX idx_daily_goals_user_date ON daily_goals(user_id, date);

-- RLS Policy
ALTER TABLE daily_goals ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own goals" ON daily_goals
    FOR SELECT USING (auth.uid() = user_id);
```

### 12. `score_predictions`

Score prediction history.

```sql
CREATE TABLE score_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    exam_type_id UUID NOT NULL REFERENCES exam_types(id),
    
    predicted_score INTEGER,
    predicted_percentile DECIMAL(5,2),
    confidence_low INTEGER,
    confidence_high INTEGER,
    
    -- Breakdown
    subject_scores JSONB, -- {"physics": 85, "chemistry": 78, "math": 92}
    
    -- Model info
    model_version VARCHAR(50),
    features_used JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_predictions_user ON score_predictions(user_id, created_at);

-- RLS Policy
ALTER TABLE score_predictions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own predictions" ON score_predictions
    FOR SELECT USING (auth.uid() = user_id);
```

---

## 🔄 Triggers

### Update timestamps

```sql
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_timestamp
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_profiles_timestamp
    BEFORE UPDATE ON user_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_mastery_timestamp
    BEFORE UPDATE ON mastery_states
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
```

### Update question statistics

```sql
CREATE OR REPLACE FUNCTION update_question_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE questions
    SET 
        total_attempts = total_attempts + 1,
        correct_attempts = correct_attempts + CASE WHEN NEW.is_correct THEN 1 ELSE 0 END,
        updated_at = NOW()
    WHERE id = NEW.question_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER on_attempt_insert
    AFTER INSERT ON attempts
    FOR EACH ROW
    EXECUTE FUNCTION update_question_stats();
```

---

## 📊 Views

### User dashboard stats

```sql
CREATE VIEW user_dashboard_stats AS
SELECT 
    u.id as user_id,
    up.streak_count,
    up.total_xp,
    up.current_level,
    COUNT(DISTINCT a.id) as total_attempts_today,
    COUNT(DISTINCT CASE WHEN a.is_correct THEN a.id END) as correct_today,
    COALESCE(
        (SELECT predicted_score FROM score_predictions sp 
         WHERE sp.user_id = u.id 
         ORDER BY created_at DESC LIMIT 1), 
        0
    ) as latest_predicted_score
FROM users u
LEFT JOIN user_profiles up ON u.id = up.user_id
LEFT JOIN attempts a ON u.id = a.user_id AND DATE(a.created_at) = CURRENT_DATE
GROUP BY u.id, up.streak_count, up.total_xp, up.current_level;
```

### Topic leaderboard

```sql
CREATE VIEW topic_leaderboard AS
SELECT 
    ms.topic,
    ms.user_id,
    u.name,
    ms.elo_rating,
    ms.mastery_probability,
    RANK() OVER (PARTITION BY ms.topic ORDER BY ms.elo_rating DESC) as rank
FROM mastery_states ms
JOIN users u ON ms.user_id = u.id
WHERE ms.total_attempts >= 10;
```

---

## 🚀 Initial Data

### JEE Topics

```sql
-- Physics topics
INSERT INTO topics (exam_type_id, subject, topic_name, order_index) VALUES
((SELECT id FROM exam_types WHERE name = 'jee_main'), 'Physics', 'Mechanics', 1),
((SELECT id FROM exam_types WHERE name = 'jee_main'), 'Physics', 'Thermodynamics', 2),
((SELECT id FROM exam_types WHERE name = 'jee_main'), 'Physics', 'Electromagnetism', 3),
((SELECT id FROM exam_types WHERE name = 'jee_main'), 'Physics', 'Optics', 4),
((SELECT id FROM exam_types WHERE name = 'jee_main'), 'Physics', 'Modern Physics', 5);

-- Chemistry topics
INSERT INTO topics (exam_type_id, subject, topic_name, order_index) VALUES
((SELECT id FROM exam_types WHERE name = 'jee_main'), 'Chemistry', 'Physical Chemistry', 1),
((SELECT id FROM exam_types WHERE name = 'jee_main'), 'Chemistry', 'Organic Chemistry', 2),
((SELECT id FROM exam_types WHERE name = 'jee_main'), 'Chemistry', 'Inorganic Chemistry', 3);

-- Mathematics topics
INSERT INTO topics (exam_type_id, subject, topic_name, order_index) VALUES
((SELECT id FROM exam_types WHERE name = 'jee_main'), 'Mathematics', 'Algebra', 1),
((SELECT id FROM exam_types WHERE name = 'jee_main'), 'Mathematics', 'Calculus', 2),
((SELECT id FROM exam_types WHERE name = 'jee_main'), 'Mathematics', 'Coordinate Geometry', 3),
((SELECT id FROM exam_types WHERE name = 'jee_main'), 'Mathematics', 'Trigonometry', 4),
((SELECT id FROM exam_types WHERE name = 'jee_main'), 'Mathematics', 'Probability & Statistics', 5);
```

---

**Last Updated**: November 2025
