# 🧠 ExamForge AI - Mastery Engine Specification

## Overview

The Mastery Engine is the brain of ExamForge AI, responsible for tracking student knowledge, predicting scores, and selecting optimal questions. It combines multiple algorithms to create a truly adaptive learning experience.

---

## 📊 Core Components

### 1. Knowledge State Model

Each student has a personalized "knowledge state" that tracks their mastery across all topics.

```
┌─────────────────────────────────────────────────────────────────┐
│                     STUDENT KNOWLEDGE STATE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    PHYSICS                               │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌──────────┐ │    │
│  │  │ Mechanics │ │ Thermo    │ │ E&M       │ │ Optics   │ │    │
│  │  │ 85%  ████ │ │ 72% ███   │ │ 65% ██    │ │ 78% ███  │ │    │
│  │  └───────────┘ └───────────┘ └───────────┘ └──────────┘ │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   CHEMISTRY                              │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐              │    │
│  │  │ Physical  │ │ Organic   │ │ Inorganic │              │    │
│  │  │ 68%  ██   │ │ 55% ██    │ │ 70% ███   │              │    │
│  │  └───────────┘ └───────────┘ └───────────┘              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  MATHEMATICS                             │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌──────────┐ │    │
│  │  │ Algebra   │ │ Calculus  │ │ Coord Geo │ │ Trig     │ │    │
│  │  │ 82%  ████ │ │ 75% ███   │ │ 60% ██    │ │ 88% ████ │ │    │
│  │  └───────────┘ └───────────┘ └───────────┘ └──────────┘ │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Algorithm 1: Bayesian Knowledge Tracing (BKT)

### Theory

BKT models knowledge as a hidden Markov model with four parameters:

| Parameter | Symbol | Description | Default |
|-----------|--------|-------------|---------|
| Prior Knowledge | P(L₀) | Probability student knew topic before first practice | 0.0 |
| Learn Rate | P(T) | Probability of learning after each practice | 0.1 |
| Slip | P(S) | Probability of wrong answer despite knowing | 0.1 |
| Guess | P(G) | Probability of right answer despite not knowing | 0.25 |

### Update Equations

After each attempt, update the knowledge probability:

```python
def update_knowledge_state(p_known: float, is_correct: bool, params: BKTParams) -> float:
    """
    Update P(L) after an attempt using BKT.
    
    Args:
        p_known: Current probability of knowing the skill
        is_correct: Whether the student answered correctly
        params: BKT parameters (p_guess, p_slip, p_transit)
    
    Returns:
        Updated probability of knowing the skill
    """
    p_g = params.p_guess  # Probability of guessing correctly
    p_s = params.p_slip   # Probability of slipping (knowing but wrong)
    p_t = params.p_transit  # Probability of learning
    
    if is_correct:
        # P(L|correct) using Bayes' rule
        p_correct_given_known = 1 - p_s
        p_correct_given_unknown = p_g
        p_correct = p_known * p_correct_given_known + (1 - p_known) * p_correct_given_unknown
        
        p_known_given_correct = (p_known * p_correct_given_known) / p_correct
    else:
        # P(L|incorrect) using Bayes' rule
        p_incorrect_given_known = p_s
        p_incorrect_given_unknown = 1 - p_g
        p_incorrect = p_known * p_incorrect_given_known + (1 - p_known) * p_incorrect_given_unknown
        
        p_known_given_correct = (p_known * p_incorrect_given_known) / p_incorrect
    
    # Apply learning transition
    p_learned = p_known_given_correct + (1 - p_known_given_correct) * p_t
    
    return p_learned
```

### Confidence Interval

We also track uncertainty using a beta distribution:

```python
def calculate_confidence_interval(attempts: int, correct: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for mastery probability.
    Uses Beta distribution with Jeffreys prior.
    """
    from scipy.stats import beta
    
    alpha = correct + 0.5
    beta_param = (attempts - correct) + 0.5
    
    lower = beta.ppf((1 - confidence) / 2, alpha, beta_param)
    upper = beta.ppf((1 + confidence) / 2, alpha, beta_param)
    
    return lower, upper
```

---

## 🎯 Algorithm 2: Elo Rating System

### For Students

Each student has an Elo rating per topic that reflects their skill level.

```python
def update_student_elo(
    student_rating: int,
    question_rating: int,
    is_correct: bool,
    k_factor: int = 32
) -> int:
    """
    Update student Elo rating after attempting a question.
    
    Args:
        student_rating: Current student Elo in this topic
        question_rating: Difficulty rating of the question
        is_correct: Whether the student answered correctly
        k_factor: Learning rate (higher = faster adjustment)
    
    Returns:
        New student Elo rating
    """
    # Expected probability of correct answer
    expected = 1 / (1 + 10 ** ((question_rating - student_rating) / 400))
    
    # Actual outcome (1 for correct, 0 for incorrect)
    actual = 1.0 if is_correct else 0.0
    
    # Update rating
    new_rating = student_rating + k_factor * (actual - expected)
    
    return round(new_rating)
```

### For Questions

Questions also have dynamic difficulty ratings that update based on student performance:

```python
def update_question_elo(
    question_rating: int,
    student_rating: int,
    is_correct: bool,
    k_factor: int = 16  # Lower K for questions (more stable)
) -> int:
    """
    Update question difficulty rating based on student performance.
    """
    expected_correct = 1 / (1 + 10 ** ((question_rating - student_rating) / 400))
    actual = 1.0 if is_correct else 0.0
    
    # Inverse update: if high-rated student gets it right, question gets easier rating
    new_rating = question_rating - k_factor * (actual - expected_correct)
    
    return round(new_rating)
```

### Elo Brackets

| Elo Range | Level Name | Description |
|-----------|------------|-------------|
| < 800 | Beginner | Just starting out |
| 800-1000 | Developing | Basic understanding |
| 1000-1200 | Intermediate | Solid fundamentals |
| 1200-1400 | Proficient | Good problem-solving |
| 1400-1600 | Advanced | Strong skills |
| 1600-1800 | Expert | Top 10% level |
| > 1800 | Master | Competition level |

---

## 🎯 Algorithm 3: Spaced Repetition (SM-2)

### Review Scheduling

Based on SuperMemo's SM-2 algorithm, adapted for exam prep:

```python
def calculate_next_review(
    quality: int,  # 0-5 rating of response quality
    previous_interval: int,  # Days since last review
    ease_factor: float  # Current ease (default 2.5)
) -> Tuple[int, float]:
    """
    Calculate next review interval and updated ease factor.
    
    Args:
        quality: 0=complete blackout, 5=perfect response
        previous_interval: Days since last review (0 for new items)
        ease_factor: Current ease factor (min 1.3)
    
    Returns:
        (next_interval_days, new_ease_factor)
    """
    # Update ease factor
    new_ease = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_ease = max(1.3, new_ease)  # Minimum ease
    
    if quality < 3:
        # Failed - reset to 1 day
        return 1, new_ease
    
    # Calculate next interval
    if previous_interval == 0:
        next_interval = 1
    elif previous_interval == 1:
        next_interval = 6
    else:
        next_interval = round(previous_interval * new_ease)
    
    # Cap at 180 days for exam prep (we want regular review)
    next_interval = min(180, next_interval)
    
    return next_interval, new_ease
```

### Quality Rating Mapping

| Attempt Outcome | Quality Score |
|-----------------|---------------|
| Correct, no hints, <30s | 5 |
| Correct, no hints, >30s | 4 |
| Correct, with hint | 3 |
| Incorrect, close answer | 2 |
| Incorrect, wrong approach | 1 |
| Complete blackout | 0 |

---

## 🎯 Algorithm 4: Optimal Question Selection

### Zone of Proximal Development (ZPD)

Select questions that are challenging but achievable (70-85% expected success rate):

```python
def select_next_question(
    student: StudentState,
    available_questions: List[Question],
    recent_topics: List[str],
    session_context: SessionContext
) -> Question:
    """
    Select the optimal next question for learning.
    
    Balances:
    1. Appropriate difficulty (ZPD)
    2. Topic interleaving
    3. Spaced repetition priorities
    4. Weak area focus
    5. Time constraints
    """
    scored_questions = []
    
    for q in available_questions:
        score = 0.0
        
        # 1. Difficulty match (40% weight)
        expected_success = calculate_expected_success(
            student.elo_for_topic(q.topic),
            q.elo_rating
        )
        # Optimal range: 70-85% success probability
        if 0.70 <= expected_success <= 0.85:
            difficulty_score = 1.0
        elif 0.60 <= expected_success < 0.70 or 0.85 < expected_success <= 0.90:
            difficulty_score = 0.7
        else:
            difficulty_score = 0.3
        score += 0.40 * difficulty_score
        
        # 2. Interleaving bonus (20% weight)
        if q.topic not in recent_topics[-3:]:
            score += 0.20
        
        # 3. Spaced repetition priority (25% weight)
        days_overdue = (datetime.now() - student.next_review_for(q.topic)).days
        if days_overdue > 0:
            sr_score = min(1.0, days_overdue / 7)  # Max priority after 7 days overdue
            score += 0.25 * sr_score
        
        # 4. Weak area focus (15% weight)
        if q.topic in student.weak_areas[:5]:
            score += 0.15
        
        scored_questions.append((q, score))
    
    # Sort by score and add some randomness to top choices
    scored_questions.sort(key=lambda x: x[1], reverse=True)
    top_candidates = scored_questions[:5]
    
    # Weighted random selection from top candidates
    weights = [s for _, s in top_candidates]
    selected = random.choices([q for q, _ in top_candidates], weights=weights, k=1)[0]
    
    return selected
```

---

## 📈 Score Prediction Model

### Features Used

```python
PREDICTION_FEATURES = [
    # Core metrics
    'overall_accuracy',  # Weighted by recency
    'avg_time_per_question',
    'total_questions_attempted',
    'total_practice_hours',
    
    # Subject-specific
    'physics_elo',
    'chemistry_elo', 
    'mathematics_elo',
    'physics_accuracy_recent',
    'chemistry_accuracy_recent',
    'mathematics_accuracy_recent',
    
    # Behavioral
    'streak_days',
    'consistency_score',  # Regular practice vs cramming
    'hint_usage_rate',
    'time_to_exam_days',
    
    # Mock test performance
    'avg_mock_score',
    'mock_score_trend',  # Improving or declining
    'mock_time_management_score',
    
    # Topic coverage
    'syllabus_coverage_percent',
    'weak_topics_count',
    'mastered_topics_count'
]
```

### Prediction Algorithm

```python
def predict_jee_score(student: StudentState) -> ScorePrediction:
    """
    Predict expected JEE score using ensemble model.
    
    Returns predicted score, percentile, and confidence interval.
    """
    features = extract_features(student)
    
    # Ensemble of models
    predictions = [
        linear_model.predict(features),
        gradient_boost_model.predict(features),
        neural_net_model.predict(features)
    ]
    
    # Weighted average (neural net gets higher weight with more data)
    if student.total_attempts > 500:
        weights = [0.2, 0.3, 0.5]
    else:
        weights = [0.4, 0.4, 0.2]
    
    predicted_score = sum(p * w for p, w in zip(predictions, weights))
    
    # Calculate confidence interval based on prediction variance
    variance = np.var(predictions)
    std_error = np.sqrt(variance)
    
    confidence_low = predicted_score - 1.96 * std_error
    confidence_high = predicted_score + 1.96 * std_error
    
    # Convert to percentile using historical JEE data
    percentile = score_to_percentile(predicted_score)
    
    return ScorePrediction(
        score=round(predicted_score),
        percentile=round(percentile, 1),
        confidence_low=round(confidence_low),
        confidence_high=round(confidence_high),
        subject_breakdown={
            'physics': predict_subject_score(student, 'physics'),
            'chemistry': predict_subject_score(student, 'chemistry'),
            'mathematics': predict_subject_score(student, 'mathematics')
        }
    )
```

### Score Timeline Visualization

```
Predicted Score Over Time
                                                          Target: 250
280 ┤                                                    ╭────────
260 ┤                                              ╭─────╯
240 ┤                                        ╭─────╯
220 ┤                                  ╭─────╯
200 ┤                            ╭─────╯
180 ┤                      ╭─────╯
160 ┤                ╭─────╯ ← You are here
140 ┤          ╭─────╯
120 ┤    ╭─────╯
100 ┼────╯
    └─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬
        Week 1    3     5     7     9    11    13    15    17   Exam
```

---

## 🔄 Mastery State Update Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Student   │────▶│   Submit    │────▶│   Validate  │
│   Answers   │     │   Answer    │     │   Response  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
       ┌───────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    UPDATE MASTERY STATE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Update     │  │  Update     │  │  Calculate Next     │ │
│  │  BKT P(L)   │  │  Elo Rating │  │  Review Date (SR)   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │           │
│         └────────────────┴─────────────────────┘           │
│                          │                                 │
│                          ▼                                 │
│              ┌─────────────────────┐                       │
│              │  Persist to DB      │                       │
│              │  mastery_states     │                       │
│              └──────────┬──────────┘                       │
│                         │                                  │
└─────────────────────────┼──────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │  Update Score       │
              │  Prediction (async) │
              └─────────────────────┘
```

---

## 📊 Weak Area Detection

```python
def identify_weak_areas(student: StudentState, n: int = 5) -> List[WeakArea]:
    """
    Identify the student's weakest topics for focused practice.
    
    Considers:
    1. Low mastery probability
    2. High JEE weightage
    3. Prerequisite for other topics
    4. Recent decline in performance
    """
    weak_areas = []
    
    for topic_state in student.mastery_states:
        weakness_score = 0.0
        
        # Low mastery (60% weight)
        if topic_state.mastery_probability < 0.7:
            weakness_score += 0.6 * (1 - topic_state.mastery_probability)
        
        # High JEE weightage (20% weight)
        topic_weight = get_jee_topic_weight(topic_state.topic)
        weakness_score += 0.2 * topic_weight
        
        # Is prerequisite (10% weight)
        dependent_topics = get_dependent_topics(topic_state.topic)
        if dependent_topics:
            weakness_score += 0.1
        
        # Recent decline (10% weight)
        recent_trend = calculate_recent_trend(topic_state)
        if recent_trend < 0:
            weakness_score += 0.1 * abs(recent_trend)
        
        weak_areas.append(WeakArea(
            topic=topic_state.topic,
            mastery=topic_state.mastery_probability,
            weakness_score=weakness_score,
            recommended_practice=generate_practice_plan(topic_state)
        ))
    
    # Sort by weakness score and return top n
    weak_areas.sort(key=lambda x: x.weakness_score, reverse=True)
    return weak_areas[:n]
```

---

## 🎮 Gamification Integration

### XP Calculation

```python
def calculate_xp_earned(
    is_correct: bool,
    question_difficulty: int,
    time_taken: int,
    hint_used: bool,
    streak_multiplier: float
) -> int:
    """
    Calculate XP earned for an attempt.
    """
    base_xp = {1: 5, 2: 10, 3: 15, 4: 25, 5: 40}[question_difficulty]
    
    if not is_correct:
        return round(base_xp * 0.1)  # Participation XP
    
    xp = base_xp
    
    # Speed bonus (up to 50%)
    expected_time = {1: 60, 2: 90, 3: 120, 4: 180, 5: 240}[question_difficulty]
    if time_taken < expected_time * 0.5:
        xp *= 1.5
    elif time_taken < expected_time:
        xp *= 1.2
    
    # Hint penalty
    if hint_used:
        xp *= 0.7
    
    # Streak multiplier
    xp *= streak_multiplier
    
    return round(xp)
```

### Level Progression

| Level | XP Required | Cumulative | Title |
|-------|-------------|------------|-------|
| 1 | 0 | 0 | Novice |
| 5 | 500 | 1,000 | Learner |
| 10 | 1,000 | 5,500 | Student |
| 15 | 2,000 | 15,500 | Scholar |
| 20 | 3,000 | 30,500 | Expert |
| 25 | 5,000 | 55,500 | Master |
| 30 | 8,000 | 95,500 | Grandmaster |

---

## 🔧 Implementation Notes

### Performance Considerations

1. **Batch Updates**: Update predictions in background, not real-time
2. **Caching**: Cache question pool filtered by topic/difficulty
3. **Incremental Updates**: Don't recalculate everything; update incrementally

### Database Indexes

```sql
-- Critical for question selection
CREATE INDEX idx_questions_topic_elo ON questions(topic, elo_rating);

-- Critical for spaced repetition
CREATE INDEX idx_mastery_next_review ON mastery_states(user_id, next_review_at);

-- For weak area detection
CREATE INDEX idx_mastery_probability ON mastery_states(user_id, mastery_probability);
```

### API Endpoints

```python
# Mastery Engine API
GET  /mastery/state/{user_id}          # Full knowledge state
GET  /mastery/weak-areas/{user_id}     # Top 5 weak areas
GET  /mastery/predictions/{user_id}    # Score predictions
POST /mastery/update                   # Update after attempt
GET  /mastery/next-question            # Get optimal next question
GET  /mastery/study-plan/{user_id}     # Personalized study plan
```

---

**Last Updated**: November 2025
