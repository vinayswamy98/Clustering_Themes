# 🏗️ ExamForge AI - System Architecture

## Overview

ExamForge AI follows a modern microservices architecture with clear separation between the presentation layer (mobile/web), business logic layer (API), and data layer (database + AI).

---

## 🎨 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Mobile App     │  │   Web App       │  │  Admin Panel    │  │
│  │  (React Native) │  │   (Next.js)     │  │  (React)        │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└───────────┼────────────────────┼────────────────────┼───────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                         API GATEWAY                              │
│                    (Rate Limiting, Auth, Logging)                │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────┐
│                        BUSINESS LOGIC LAYER                      │
├────────────────────────────────┴────────────────────────────────┤
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │  User      │  │  Question  │  │  Mastery   │  │  AI Tutor  │ │
│  │  Service   │  │  Service   │  │  Engine    │  │  Service   │ │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘ │
│        │               │               │               │        │
│  ┌─────┴───────────────┴───────────────┴───────────────┴─────┐  │
│  │                    Message Queue (Redis)                   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────┐
│                          DATA LAYER                              │
├────────────────────────────────┴────────────────────────────────┤
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │ PostgreSQL │  │   Redis    │  │  ChromaDB  │  │  S3/CDN    │ │
│  │ (Supabase) │  │  (Cache)   │  │  (Vectors) │  │  (Media)   │ │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────┐
│                        EXTERNAL SERVICES                         │
├────────────────────────────────┴────────────────────────────────┤
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │  OpenAI    │  │  Stripe    │  │  Firebase  │  │  Sentry    │ │
│  │  (LLM)     │  │  (Payments)│  │  (Push)    │  │  (Errors)  │ │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Core Data Flow

### Question Practice Flow

```
┌─────────┐    ┌─────────┐    ┌─────────────┐    ┌──────────────┐
│  User   │───▶│  API    │───▶│   Mastery   │───▶│   Question   │
│  App    │    │  Gateway│    │   Engine    │    │   Selection  │
└─────────┘    └─────────┘    └─────────────┘    └──────┬───────┘
                                                        │
                                                        ▼
┌─────────┐    ┌─────────┐    ┌─────────────┐    ┌──────────────┐
│ Display │◀───│  Format │◀───│   Retrieve  │◀───│   Question   │
│ Question│    │  Response│   │   Question  │    │   Database   │
└────┬────┘    └─────────┘    └─────────────┘    └──────────────┘
     │
     ▼
┌─────────┐    ┌─────────┐    ┌─────────────┐    ┌──────────────┐
│  User   │───▶│  Submit │───▶│   Validate  │───▶│   Update     │
│  Answer │    │  Answer │    │   Answer    │    │   Mastery    │
└─────────┘    └─────────┘    └─────────────┘    └──────────────┘
```

### AI Tutor Flow

```
┌─────────┐    ┌─────────┐    ┌─────────────┐
│  User   │───▶│  "Help  │───▶│   Context   │
│  Request│    │  Me"    │    │   Builder   │
└─────────┘    └─────────┘    └──────┬──────┘
                                     │
     ┌───────────────────────────────┘
     ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Retrieve  │───▶│   Generate  │───▶│   LLM API   │
│   History   │    │   Prompt    │    │   (GPT-4o)  │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             │
     ┌───────────────────────────────────────┘
     ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Cache     │───▶│   Format    │───▶│   Display   │
│   Response  │    │   Response  │    │   to User   │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## 🔧 Service Specifications

### 1. User Service
**Responsibility**: Authentication, profiles, preferences

```python
Endpoints:
  POST /auth/register
  POST /auth/login
  POST /auth/refresh
  GET  /users/me
  PUT  /users/me
  GET  /users/me/stats
```

### 2. Question Service
**Responsibility**: Question CRUD, search, tagging

```python
Endpoints:
  GET  /questions                    # List with filters
  GET  /questions/{id}              # Single question
  GET  /questions/recommended       # Personalized recommendations
  POST /questions/attempt           # Submit answer
  GET  /questions/search            # Full-text search
```

### 3. Mastery Engine
**Responsibility**: Skill tracking, predictions, recommendations

```python
Endpoints:
  GET  /mastery/state               # Current knowledge state
  GET  /mastery/predictions         # Score predictions
  GET  /mastery/weak-areas          # Areas needing work
  GET  /mastery/study-plan          # Recommended study path
  POST /mastery/update              # Update after attempt
```

### 4. AI Tutor Service
**Responsibility**: LLM-powered explanations and hints

```python
Endpoints:
  POST /tutor/explain               # Explain a concept
  POST /tutor/hint                  # Get a hint for current question
  POST /tutor/solve-step-by-step    # Detailed solution walkthrough
  POST /tutor/chat                  # Free-form tutoring conversation
```

---

## 💾 Database Architecture

### Core Tables

```sql
-- Users
users
├── id (UUID, PK)
├── email
├── name
├── exam_type (JEE, NEET, SAT, etc.)
├── target_date
├── current_level
├── streak_count
├── total_xp
└── created_at

-- Questions
questions
├── id (UUID, PK)
├── exam_type
├── subject (Physics, Chemistry, Math)
├── topic
├── subtopic
├── difficulty (1-5)
├── elo_rating
├── question_text
├── question_image_url
├── options (JSONB)
├── correct_answer
├── solution_text
├── solution_image_url
├── tags (ARRAY)
├── year (previous year question reference)
└── created_at

-- Attempts
attempts
├── id (UUID, PK)
├── user_id (FK)
├── question_id (FK)
├── selected_answer
├── is_correct
├── time_taken_seconds
├── hint_used
├── created_at
└── session_id

-- Mastery State
mastery_states
├── id (UUID, PK)
├── user_id (FK)
├── topic
├── subtopic
├── elo_rating
├── confidence_level (0-1)
├── total_attempts
├── correct_attempts
├── last_practiced_at
└── next_review_at (spaced repetition)

-- Sessions
study_sessions
├── id (UUID, PK)
├── user_id (FK)
├── type (practice, mock_test, review)
├── started_at
├── ended_at
├── questions_attempted
├── questions_correct
├── xp_earned
└── topics_covered (ARRAY)
```

### Indexes

```sql
-- Performance indexes
CREATE INDEX idx_questions_exam_topic ON questions(exam_type, topic);
CREATE INDEX idx_attempts_user_date ON attempts(user_id, created_at);
CREATE INDEX idx_mastery_user_topic ON mastery_states(user_id, topic);
CREATE INDEX idx_mastery_next_review ON mastery_states(next_review_at);
```

---

## 🔒 Security Architecture

### Authentication Flow

```
┌─────────┐    ┌─────────┐    ┌─────────────┐    ┌──────────────┐
│  User   │───▶│  Login  │───▶│   Supabase  │───▶│   JWT Token  │
│         │    │  Screen │    │   Auth      │    │   Returned   │
└─────────┘    └─────────┘    └─────────────┘    └──────┬───────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Secure Token Storage                          │
│                  (Keychain/Secure Storage)                       │
└─────────────────────────────────────────────────────────────────┘
```

### Security Measures
1. **JWT tokens** with 1-hour expiry + refresh tokens
2. **Row-Level Security (RLS)** in Supabase
3. **API rate limiting** (100 req/min default)
4. **Input validation** on all endpoints
5. **Encrypted data at rest** (Supabase default)
6. **HTTPS only** in production

---

## 📊 Caching Strategy

### Cache Layers

| Layer | Storage | TTL | Use Case |
|-------|---------|-----|----------|
| L1 | In-memory | 5 min | User session, current question |
| L2 | Redis | 1 hour | Question bank, leaderboards |
| L3 | CDN | 24 hours | Static assets, images |

### Cache Keys

```
user:{user_id}:mastery           # User's mastery state
user:{user_id}:daily_goal        # Today's goal progress
question:{question_id}           # Question details
leaderboard:{exam_type}:daily    # Daily leaderboard
ai_response:{hash}               # Cached AI explanations
```

---

## 🚀 Deployment Architecture

### Production Environment

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLOUDFLARE                               │
│                    (CDN + DDoS Protection)                       │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────┐
│                            RAILWAY                               │
├────────────────────────────────┴────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐ │
│  │   API Server   │  │   API Server   │  │   Worker Server    │ │
│  │   (Instance 1) │  │   (Instance 2) │  │   (Background)     │ │
│  └────────────────┘  └────────────────┘  └────────────────────┘ │
│                              │                                   │
│                   ┌──────────┴──────────┐                       │
│                   │     Load Balancer   │                       │
│                   └─────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────┐
│                           SUPABASE                               │
├────────────────────────────────┴────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐ │
│  │   PostgreSQL   │  │     Auth       │  │     Storage        │ │
│  │   (Primary)    │  │   (Supabase)   │  │   (Images/Files)   │ │
│  └────────────────┘  └────────────────┘  └────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### CI/CD Pipeline

```yaml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: pytest

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: railwayapp/deploy@v1
        with:
          railway_token: ${{ secrets.RAILWAY_TOKEN }}
```

---

## 📈 Monitoring & Observability

### Metrics

| Metric | Tool | Alert Threshold |
|--------|------|-----------------|
| API Latency | Railway Metrics | > 500ms |
| Error Rate | Sentry | > 1% |
| Database Connections | Supabase | > 80% pool |
| AI API Costs | Custom Dashboard | > $50/day |
| Active Users | Mixpanel | N/A |

### Logging

```python
# Structured logging format
{
    "timestamp": "2025-11-30T10:00:00Z",
    "level": "INFO",
    "service": "mastery-engine",
    "user_id": "uuid-here",
    "action": "update_mastery",
    "topic": "thermodynamics",
    "old_rating": 1200,
    "new_rating": 1215,
    "duration_ms": 45
}
```

---

## 🔄 Scalability Considerations

### Horizontal Scaling

- **API Servers**: Stateless, can add instances freely
- **Database**: Supabase handles read replicas
- **Redis**: Can cluster for larger workloads
- **AI**: Queue-based processing for high load

### Estimated Capacity

| Users | API Instances | Database Tier | Monthly Cost |
|-------|---------------|---------------|--------------|
| 1K | 1 | Free | ~$50 |
| 10K | 2 | Pro | ~$200 |
| 100K | 4 | Enterprise | ~$1,000 |
| 1M | 8+ | Custom | ~$5,000+ |

---

**Last Updated**: November 2025
