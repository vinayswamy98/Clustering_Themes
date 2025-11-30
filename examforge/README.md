# 🚀 ExamForge AI - Ultimate Personalized Exam Preparation Platform

> *The app that ends exam anxiety forever.*

ExamForge AI is a revolutionary personalized exam-preparation platform that adapts in real-time to every student, using cutting-edge AI and learning science to maximize score improvement in minimum time.

## 🎯 Vision

**100% personalization** - No two students ever get the same experience.

## 📚 Target Exams

### Primary (MVP)
- **JEE Main + Advanced 2026** (India) - One of the hardest and most lucrative markets

### Expansion Phase
- SAT, ACT (US)
- NEET (India - Medical)
- GRE, GMAT (Graduate)
- IELTS, TOEFL (Language)
- UPSC (Indian Civil Services)
- USMLE, Bar Exam (Professional)

## 🧠 Core Principles

1. **Maximize Score Improvement** - In minimum time
2. **100% Personalization** - Adaptive to individual learning patterns
3. **Evidence-Based Learning Science**
   - Spaced Repetition
   - Active Recall
   - Interleaving
   - Metacognition
   - Desirable Difficulty
   - Growth Mindset
4. **Brutally Addictive UX** - Duolingo-level retention + peak flow state
5. **Exam-Specific Mastery** - Perfect mimicry of real exam format, timing, scoring, and traps

## 📂 Project Structure

```
examforge/
├── README.md                    # This file
├── docs/
│   ├── MVP_SPEC.md             # 8-12 week MVP launch plan
│   ├── ARCHITECTURE.md         # System architecture
│   ├── DATABASE_SCHEMA.md      # Database design
│   ├── AI_TUTOR_PROMPTS.md     # AI tutor prompt engineering
│   ├── MASTERY_ENGINE.md       # Mastery tracking algorithm
│   ├── GAMIFICATION.md         # Gamification system design
│   ├── MONETIZATION.md         # Monetization strategy
│   └── COMPETITOR_ANALYSIS.md  # Competitive differentiation
├── api/                         # Backend API (FastAPI/Node.js)
│   ├── requirements.txt
│   └── src/
├── mobile/                      # React Native + Expo app
│   └── package.json
└── web/                         # Web dashboard
    └── package.json
```

## 🔧 Tech Stack

### Backend
- **API**: FastAPI (Python) or Node.js
- **Database**: PostgreSQL via Supabase
- **Vector DB**: ChromaDB/Pinecone for RAG
- **Cache**: Redis

### AI/ML
- **LLM**: OpenAI GPT-4o / Anthropic Claude / Grok
- **Embeddings**: Sentence Transformers
- **Adaptive Algorithm**: Bayesian Knowledge Tracing + Elo Rating

### Frontend
- **Mobile**: React Native + Expo
- **Web**: Next.js / React
- **State**: Redux/Zustand

### Infrastructure
- **Hosting**: Vercel / Railway / AWS
- **CDN**: Cloudflare
- **Analytics**: Mixpanel / Amplitude

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm or yarn

### Running the API Server

```bash
# Navigate to the API directory
cd examforge/api

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your configuration
cat > .env << EOF
DEBUG=true
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
OPENAI_API_KEY=your_openai_key
EOF

# Run the development server
uvicorn src.main:app --reload --port 8000

# API will be available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Running the Mobile App (React Native + Expo)

```bash
# Navigate to the mobile directory
cd examforge/mobile

# Install dependencies
npm install

# Start the Expo development server
npm start

# Then scan the QR code with Expo Go app on your phone
# Or press 'a' for Android emulator, 'i' for iOS simulator
```

### Running the Web Dashboard (Next.js)

```bash
# Navigate to the web directory
cd examforge/web

# Install dependencies
npm install

# Start the development server
npm run dev

# Web app will be available at http://localhost:3000
```

### Quick API Test

Once the API is running, test it with:

```bash
# Health check
curl http://localhost:8000/health

# View API documentation
open http://localhost:8000/docs
```

See [MVP_SPEC.md](docs/MVP_SPEC.md) for the detailed launch plan.

## 📊 Competitive Advantages

| Feature | ExamForge AI | Duolingo | Anki | Khan Academy | Byju's |
|---------|-------------|----------|------|--------------|--------|
| Real-time Adaptation | ✅ | ⚠️ | ❌ | ⚠️ | ⚠️ |
| AI Tutor | ✅ | ❌ | ❌ | ❌ | ⚠️ |
| Exam-Specific Format | ✅ | N/A | ❌ | ⚠️ | ✅ |
| Gamification | ✅ | ✅ | ❌ | ⚠️ | ⚠️ |
| Predicted Score | ✅ | ❌ | ❌ | ❌ | ⚠️ |
| Personalized Path | ✅ | ⚠️ | ❌ | ⚠️ | ⚠️ |

## 📈 Success Metrics

- **User Engagement**: Daily Active Users (DAU), Session Length, Streak Rate
- **Learning Outcomes**: Score Improvement, Topic Mastery Rate
- **Retention**: D1/D7/D30 Retention, Churn Rate
- **Revenue**: ARPU, LTV, CAC

## 📄 License

Proprietary - All Rights Reserved

## 📧 Contact

For inquiries, please reach out to the development team.
