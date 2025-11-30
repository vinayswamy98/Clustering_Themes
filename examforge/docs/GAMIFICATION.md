# 🎮 ExamForge AI - Gamification System

## Overview

ExamForge AI uses sophisticated gamification mechanics inspired by Duolingo, video games, and behavioral psychology to create an addictive yet effective learning experience.

---

## 🎯 Core Gamification Pillars

### 1. Progress & Mastery
### 2. Streaks & Consistency
### 3. Competition & Social
### 4. Rewards & Recognition
### 5. Personalization & Autonomy

---

## 🔥 Streak System

### Daily Streak

The streak is the #1 retention driver. Protect it fiercely.

```
┌─────────────────────────────────────────────────────────────────┐
│                      🔥 15 Day Streak!                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  M   T   W   T   F   S   S   M   T   W   T   F   S   S   M     │
│  ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓   🔥    │
│  4   5   6   7   8   9  10  11  12  13  14  15  16  17  18     │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Complete today's goal to extend your streak!           │   │
│  │  Progress: ████████████░░░░░░░░ 60% (12/20 questions)   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Streak Mechanics

| Feature | Description |
|---------|-------------|
| **Minimum Activity** | Answer 5 questions OR 10 minutes of practice |
| **Timezone Aware** | Resets at midnight in user's timezone |
| **Streak Freeze** | 2 free per month (premium: 5), protects streak for 1 day |
| **Streak Repair** | Gems can restore streak within 24 hours |
| **Streak Milestones** | 7, 14, 30, 60, 100, 365 days |
| **Streak Multiplier** | XP bonus: 1x (day 1-6), 1.5x (7-29), 2x (30+) |

### Streak Protection UX

**Push Notification Schedule:**
```
6:00 PM - "Don't forget your streak! Quick 5-minute session?"
8:00 PM - "⚠️ 4 hours left to keep your streak alive!"
10:00 PM - "FINAL CALL! Your 15-day streak is at risk! 🔥"
```

**Streak Freeze Dialog:**
```
┌─────────────────────────────────────────────────────────────────┐
│                 😰 Streak in Danger!                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  You haven't practiced today and your 15-day streak            │
│  is about to end!                                               │
│                                                                 │
│  ┌───────────────────────┐   ┌───────────────────────┐         │
│  │                       │   │                       │         │
│  │   🔥 Practice Now     │   │   ❄️ Use Streak      │         │
│  │      (Recommended)    │   │      Freeze (2 left) │         │
│  │                       │   │                       │         │
│  └───────────────────────┘   └───────────────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⭐ XP & Leveling System

### XP Sources

| Action | Base XP | Notes |
|--------|---------|-------|
| Correct answer (Easy) | 5 | |
| Correct answer (Medium) | 10 | |
| Correct answer (Hard) | 20 | |
| Correct answer (Expert) | 40 | |
| Complete daily goal | 50 | Bonus |
| Complete a topic | 100 | One-time |
| Mock test completion | 200 | Per test |
| Streak milestone | 50-500 | Scales with streak |
| Achievement unlocked | 10-1000 | Varies |
| Help another student | 20 | Future social feature |

### XP Multipliers

| Condition | Multiplier |
|-----------|------------|
| Streak 7-29 days | 1.5x |
| Streak 30+ days | 2x |
| Weekend bonus | 1.2x |
| Speed bonus (<50% expected time) | 1.5x |
| No hints used | 1.2x |
| First try correct | 1.1x |

### Level Progression

```python
def xp_for_level(level: int) -> int:
    """Calculate total XP needed to reach a level."""
    if level <= 1:
        return 0
    # Exponential curve with softening at higher levels
    base = 100
    return round(base * (level ** 1.8))

# Level thresholds:
# Level 1: 0 XP
# Level 5: 1,800 XP
# Level 10: 6,300 XP
# Level 15: 13,800 XP
# Level 20: 24,500 XP
# Level 25: 38,600 XP
# Level 30: 56,200 XP
```

### Level Titles

| Level Range | Title | Badge Color |
|-------------|-------|-------------|
| 1-4 | Newcomer | Bronze |
| 5-9 | Learner | Bronze |
| 10-14 | Student | Silver |
| 15-19 | Scholar | Silver |
| 20-24 | Expert | Gold |
| 25-29 | Master | Gold |
| 30+ | Grandmaster | Diamond |

---

## 🏆 Achievement System

### Achievement Categories

#### 📚 Practice Achievements
| Achievement | Requirement | XP Reward |
|-------------|-------------|-----------|
| First Step | Answer 1 question | 10 |
| Century | Answer 100 questions | 100 |
| Thousand Club | Answer 1,000 questions | 500 |
| Practice Makes Perfect | Answer 10,000 questions | 2,000 |

#### 🔥 Streak Achievements
| Achievement | Requirement | XP Reward |
|-------------|-------------|-----------|
| Week Warrior | 7-day streak | 100 |
| Fortnight Fighter | 14-day streak | 200 |
| Monthly Master | 30-day streak | 500 |
| Unstoppable | 100-day streak | 2,000 |
| Year-Long Learner | 365-day streak | 10,000 |

#### 🎯 Mastery Achievements
| Achievement | Requirement | XP Reward |
|-------------|-------------|-----------|
| First Mastery | Master any topic | 200 |
| Physics Prodigy | 90%+ mastery in Physics | 1,000 |
| Chemistry Champion | 90%+ mastery in Chemistry | 1,000 |
| Math Maestro | 90%+ mastery in Mathematics | 1,000 |
| Triple Crown | 90%+ in all subjects | 5,000 |

#### 📈 Performance Achievements
| Achievement | Requirement | XP Reward |
|-------------|-------------|-----------|
| Perfect Session | 20 correct in a row | 200 |
| Speed Demon | Complete 10 questions in 5 min | 150 |
| Improvement Arc | Increase predicted score by 20 pts | 300 |
| Mock Master | Score 250+ on mock test | 500 |

#### 🎓 Milestone Achievements
| Achievement | Requirement | XP Reward |
|-------------|-------------|-----------|
| Getting Started | Complete onboarding | 50 |
| First Mock | Complete first mock test | 100 |
| Syllabus Scout | Attempt question from every topic | 300 |
| Target Setter | Set a score target | 25 |

### Achievement Display

```
┌─────────────────────────────────────────────────────────────────┐
│                    🏆 Achievement Unlocked!                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                         🔥                                      │
│                                                                 │
│                   WEEK WARRIOR                                  │
│             7-Day Streak Achieved!                              │
│                                                                 │
│                     +100 XP                                     │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Share Achievement                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Daily Goals

### Personalized Goals

Goals adapt to user behavior:

```python
def calculate_daily_goal(user: User) -> DailyGoal:
    """
    Calculate personalized daily goal based on:
    - Historical activity
    - Days to exam
    - Current performance
    - User preference
    """
    # Base goal from user preference
    base_minutes = user.preferred_daily_minutes  # 15, 30, 45, or 60
    
    # Adjust based on days to exam
    days_to_exam = (user.target_date - today()).days
    if days_to_exam < 30:
        urgency_multiplier = 1.5
    elif days_to_exam < 60:
        urgency_multiplier = 1.2
    else:
        urgency_multiplier = 1.0
    
    target_minutes = round(base_minutes * urgency_multiplier)
    target_questions = target_minutes * 0.5  # ~2 min per question
    target_xp = target_questions * 15  # Avg 15 XP per question
    
    return DailyGoal(
        minutes=target_minutes,
        questions=round(target_questions),
        xp=round(target_xp)
    )
```

### Goal UI

```
┌─────────────────────────────────────────────────────────────────┐
│                      Today's Goals                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📚 Questions    ████████████░░░░░░░░  60% (12/20)             │
│                                                                 │
│  ⏱️ Study Time   ██████████████░░░░░░  70% (21/30 min)         │
│                                                                 │
│  ⭐ XP Earned    ████████░░░░░░░░░░░░  40% (120/300)           │
│                                                                 │
│  ─────────────────────────────────────────────────────────     │
│                                                                 │
│  Complete all goals to earn +50 XP bonus! 🎁                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💎 Gems & Economy

### Gem Sources

| Source | Gems |
|--------|------|
| Complete daily goal | 5 |
| 7-day streak | 25 |
| 30-day streak | 100 |
| Achievement unlock | 5-50 |
| Refer a friend | 100 |
| Watch ad (free users) | 5 |
| Purchase | Varies |

### Gem Sinks

| Use | Cost |
|-----|------|
| Streak freeze | 50 gems |
| Streak repair (within 24hr) | 100 gems |
| Unlock bonus questions | 30 gems |
| Double XP boost (1 hour) | 75 gems |
| Skip diagnostic test | 200 gems |

### Premium Benefits

| Feature | Free | Premium |
|---------|------|---------|
| Daily streak freezes | 2/month | 5/month |
| Ad-free experience | ❌ | ✅ |
| Unlimited practice | Limited | ✅ |
| Detailed analytics | Basic | Full |
| AI tutor depth | Basic | Unlimited |
| Offline mode | ❌ | ✅ |
| Priority support | ❌ | ✅ |

---

## 🏅 Leaderboards

### Leaderboard Types

1. **Daily** - Resets at midnight (most competitive)
2. **Weekly** - Sunday reset
3. **All-Time** - Cumulative
4. **Topic-Specific** - Per subject/topic
5. **Friend League** - Among connections

### Ranking Leagues

Similar to Duolingo's league system:

| League | Rank Requirement | Promotion | Relegation |
|--------|------------------|-----------|------------|
| Bronze | New users | Top 20% → Silver | N/A |
| Silver | Top 20% of Bronze | Top 20% → Gold | Bottom 20% → Bronze |
| Gold | Top 20% of Silver | Top 20% → Platinum | Bottom 20% → Silver |
| Platinum | Top 20% of Gold | Top 10% → Diamond | Bottom 20% → Gold |
| Diamond | Top 10% of Platinum | N/A | Bottom 10% → Platinum |

### Leaderboard UI

```
┌─────────────────────────────────────────────────────────────────┐
│           🏆 Weekly Leaderboard - Gold League                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PROMOTION ZONE (Top 20% advance to Platinum)                   │
│  ──────────────────────────────────────────────                 │
│  🥇 #1   Priya S.        4,520 XP    ↑3                        │
│  🥈 #2   Rahul M.        4,280 XP    ↑1                        │
│  🥉 #3   Ankit K.        4,100 XP    ↓2                        │
│      #4  Sneha P.        3,890 XP    ─                         │
│  ──────────────────────────────────────────────                 │
│      ...                                                        │
│  ──────────────────────────────────────────────                 │
│  👤 #15  You             2,340 XP    ↑5                        │
│  ──────────────────────────────────────────────                 │
│      ...                                                        │
│  ──────────────────────────────────────────────                 │
│  DANGER ZONE (Bottom 20% relegated to Silver)                   │
│      #42 Vikram R.         890 XP    ↓8                        │
│                                                                 │
│  ⏰ 3 days, 4 hours until reset                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎪 Events & Challenges

### Challenge Types

1. **Daily Challenge** - Special themed question set
2. **Weekend Sprint** - 2x XP events
3. **Topic Tournament** - Compete on specific topic
4. **Mock Marathon** - Complete 3 mock tests in a week
5. **Streak Challenge** - Community goal (e.g., 1M combined streak days)

### Event Calendar

```
┌─────────────────────────────────────────────────────────────────┐
│                    📅 This Week's Events                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  MON  ┌──────────────────────────────────────────┐             │
│       │ 🧪 Chemistry Monday - 2x XP on Chemistry │             │
│       └──────────────────────────────────────────┘             │
│                                                                 │
│  WED  ┌──────────────────────────────────────────┐             │
│       │ 📐 Calculus Challenge - Win 500 gems!    │             │
│       └──────────────────────────────────────────┘             │
│                                                                 │
│  SAT  ┌──────────────────────────────────────────┐             │
│       │ 🏃 Weekend Sprint - 2x XP All Day!       │             │
│  SUN  └──────────────────────────────────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📱 Notification Strategy

### Notification Types

| Type | Timing | Content Example |
|------|--------|-----------------|
| Streak Reminder | 6 PM, 8 PM, 10 PM | "Don't lose your 15-day streak! 🔥" |
| Daily Goal | 3 PM if not started | "Start your daily practice! 5 questions to hit your goal." |
| Achievement | Immediate | "🏆 Achievement Unlocked: Week Warrior!" |
| Leaderboard | Weekly | "You moved up 5 spots! Keep going! 📈" |
| Challenge Start | Event start | "⚡ Weekend Sprint starts NOW! 2x XP awaits!" |
| Inactivity | After 3 days | "We miss you! Your Physics skills might be getting rusty 😅" |

### Notification Limits

- Max 3 per day
- Respect quiet hours (10 PM - 7 AM)
- Allow granular opt-out
- Reduce frequency for engaged users

---

## 🧠 Psychological Principles

### Applied Principles

1. **Variable Reward** - Randomized XP bonuses, surprise achievements
2. **Loss Aversion** - Streak protection mechanics, relegation fear
3. **Social Proof** - Leaderboards, friend activity
4. **Commitment & Consistency** - Daily goals, streak investment
5. **Endowed Progress** - Start users at 10% on their first goal
6. **Goal Gradient** - Increase urgency as goals approach (90% complete)
7. **Fresh Start Effect** - Weekly resets, new leagues

### Ethical Guardrails

1. **No dark patterns** - Easy unsubscribe, clear gem costs
2. **Study breaks encouraged** - Notification after 2+ hours of continuous use
3. **Health reminders** - "Take a 5-minute break! 🧘"
4. **Honest progress** - Never inflate scores or hide struggles
5. **Parental controls** - Daily time limits, spending caps

---

## 📊 Gamification Metrics

### Key Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| D1 Retention | 60% | Users returning next day |
| D7 Retention | 40% | Users returning after 7 days |
| D30 Retention | 25% | Users returning after 30 days |
| Avg Streak Length | 14 days | Mean streak before break |
| Daily Goal Completion | 50% | % users completing daily goal |
| Streak Freeze Usage | <20% | Should be emergency only |
| Premium Conversion | 5% | Free → Paid |

### A/B Tests to Run

1. Streak freeze cost (30 vs 50 vs 100 gems)
2. Daily goal difficulty (easy vs adaptive)
3. Leaderboard size (20 vs 50 vs 100 users)
4. XP multiplier values
5. Achievement rarity/difficulty

---

**Last Updated**: November 2025
