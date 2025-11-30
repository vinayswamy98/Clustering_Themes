# 🎯 ExamForge AI - MVP Specification

## 8-12 Week Launch Plan

> **Target**: JEE Main + Advanced 2026

### Executive Summary

A ruthless 10-week MVP focusing on the core adaptive learning loop with JEE-specific content. Ship fast, learn faster.

---

## 📅 Week-by-Week Breakdown

### Phase 1: Foundation (Weeks 1-3)

#### Week 1: Core Infrastructure
- [ ] Set up Supabase project (PostgreSQL + Auth)
- [ ] Initialize React Native + Expo project
- [ ] Configure FastAPI backend skeleton
- [ ] Set up CI/CD pipeline
- [ ] Design database schema (users, questions, attempts)

**Deliverables**: Project scaffolding, local dev environment, basic auth flow

#### Week 2: Question Bank Foundation
- [ ] Import 500 JEE questions (Physics, Chemistry, Math)
- [ ] Build question tagging system (topic, subtopic, difficulty)
- [ ] Create admin interface for question management
- [ ] Implement question API endpoints

**Deliverables**: Searchable question bank with 500+ questions

#### Week 3: Basic Practice Mode
- [ ] Build question display UI (math rendering with KaTeX)
- [ ] Implement answer submission flow
- [ ] Show immediate feedback (correct/incorrect + solution)
- [ ] Basic progress tracking

**Deliverables**: Students can practice questions and see solutions

---

### Phase 2: Adaptive Engine (Weeks 4-6)

#### Week 4: Mastery Tracking
- [ ] Implement Elo rating system per topic
- [ ] Build knowledge state model
- [ ] Create topic dependency graph for JEE syllabus
- [ ] Design mastery visualization

**Deliverables**: Real-time skill tracking per topic

#### Week 5: Adaptive Question Selection
- [ ] Implement question difficulty calibration
- [ ] Build "optimal challenge" algorithm (70-80% success rate target)
- [ ] Add spaced repetition for weak topics
- [ ] Create personalized daily practice sets

**Deliverables**: Personalized question recommendations

#### Week 6: AI Tutor v1
- [ ] Integrate OpenAI/Claude API
- [ ] Build solution explanation prompts
- [ ] Implement "hint" system
- [ ] Add "explain differently" feature

**Deliverables**: AI-powered explanations and hints

---

### Phase 3: Engagement & Gamification (Weeks 7-8)

#### Week 7: Core Gamification
- [ ] Implement streak system
- [ ] Build XP and level progression
- [ ] Create daily goals and challenges
- [ ] Add achievement badges

**Deliverables**: Engaging progression system

#### Week 8: Predicted Score Engine
- [ ] Build JEE score prediction model
- [ ] Create "Score Timeline" visualization
- [ ] Implement weak area identification
- [ ] Add study plan recommendations

**Deliverables**: Predicted score with improvement roadmap

---

### Phase 4: Polish & Launch (Weeks 9-10)

#### Week 9: Mock Tests
- [ ] Build timed mock test mode
- [ ] Replicate exact JEE Main format (90 questions, 3 hours)
- [ ] Implement negative marking logic
- [ ] Add detailed performance analytics

**Deliverables**: Full-length realistic mock tests

#### Week 10: Launch Prep
- [ ] Performance optimization
- [ ] Bug fixing sprint
- [ ] App store submission
- [ ] Landing page and onboarding
- [ ] Beta user acquisition (100 students)

**Deliverables**: Production-ready MVP

---

## 🎁 MVP Feature Set

### Core Features (Must Have)
1. **Practice Mode** - Unlimited adaptive practice
2. **Question Bank** - 500+ JEE questions with solutions
3. **Mastery Tracking** - Topic-wise skill visualization
4. **AI Explanations** - On-demand concept clarification
5. **Streak & XP** - Daily engagement mechanics
6. **Predicted Score** - JEE score estimation
7. **Mock Tests** - Full-length timed tests

### Deferred to V1
- Social features (leaderboards, study groups)
- Video content integration
- Parent dashboard
- Offline mode
- Multi-language support

---

## 📊 Success Criteria

### Week 10 Launch Metrics
| Metric | Target |
|--------|--------|
| Questions in bank | 500+ |
| Beta users | 100 |
| D1 Retention | 60% |
| Avg session length | 15+ min |
| Questions answered/user/day | 20+ |
| App store rating | 4.5+ |

### Technical KPIs
| Metric | Target |
|--------|--------|
| API response time | < 200ms |
| App load time | < 2s |
| Crash rate | < 0.1% |
| Uptime | 99.5% |

---

## 💰 Budget Estimate (MVP)

| Category | Cost/Month |
|----------|------------|
| Supabase Pro | $25 |
| OpenAI API | $200 |
| Expo/React Native | $0 |
| Cloud hosting | $50 |
| Domain + SSL | $20 |
| **Total** | **~$295/month** |

---

## 🚀 Post-MVP Roadmap

### V1 (Weeks 11-16)
- Expand question bank to 2000+
- Add JEE Advanced specific features
- Implement leaderboards
- Add parent dashboard

### V2 (Weeks 17-24)
- NEET exam support
- Video lessons integration
- Study groups
- Offline mode

### V3 (Weeks 25+)
- SAT/ACT expansion
- AI-generated questions
- Multi-language support
- Corporate/school licensing

---

## 🎯 Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Question quality | Partner with JEE tutors for content review |
| AI costs too high | Implement caching, use Claude for complex queries |
| User acquisition | Partner with coaching centers, influencers |
| Competition | Focus on personalization as key differentiator |
| Technical debt | Weekly code reviews, automated testing |

---

## 📱 MVP Screens

1. **Onboarding** - Goal setting, diagnostic test
2. **Home** - Daily goal, streak, recommended practice
3. **Practice** - Question view, answer, feedback
4. **Progress** - Topic mastery map, predicted score
5. **Mock Test** - Timed test interface
6. **Profile** - Settings, achievements

---

## 🔧 Technical Stack (MVP)

```yaml
Backend:
  - Framework: FastAPI (Python 3.11+)
  - Database: Supabase (PostgreSQL)
  - Auth: Supabase Auth
  - AI: OpenAI GPT-4o

Frontend:
  - Framework: React Native + Expo
  - State: Zustand
  - Styling: NativeWind (Tailwind)
  - Math: react-native-katex

DevOps:
  - CI/CD: GitHub Actions
  - Hosting: Railway (API) + Expo EAS (App)
  - Monitoring: Sentry
```

---

**Last Updated**: November 2025
