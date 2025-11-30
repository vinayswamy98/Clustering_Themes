# ExamForge AI Mobile App

React Native + Expo mobile application for ExamForge AI.

## Setup

```bash
# Install dependencies
npm install

# Start development server
npm start

# Run on iOS
npm run ios

# Run on Android
npm run android
```

## Project Structure

```
mobile/
├── App.tsx                    # App entry point
├── src/
│   ├── screens/              # Screen components
│   │   ├── HomeScreen.tsx
│   │   ├── PracticeScreen.tsx
│   │   ├── ProfileScreen.tsx
│   │   └── ...
│   ├── components/           # Reusable components
│   │   ├── QuestionCard.tsx
│   │   ├── ProgressBar.tsx
│   │   └── ...
│   ├── navigation/           # Navigation setup
│   ├── hooks/                # Custom hooks
│   ├── store/                # Zustand store
│   ├── api/                  # API client
│   ├── utils/                # Utility functions
│   └── types/                # TypeScript types
├── assets/                   # Images, fonts
└── app.json                  # Expo config
```

## Key Features

- Adaptive question practice
- AI tutor integration
- Streak and gamification
- Mock tests
- Progress tracking

## Tech Stack

- **Framework**: React Native + Expo
- **State**: Zustand
- **Styling**: NativeWind (Tailwind CSS)
- **Math**: react-native-katex
- **Navigation**: React Navigation
