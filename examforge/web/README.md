# ExamForge AI Web Dashboard

Next.js web application for ExamForge AI - admin and parent dashboards.

## Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## Project Structure

```
web/
├── app/                      # Next.js App Router
│   ├── page.tsx             # Landing page
│   ├── dashboard/           # User dashboard
│   ├── admin/               # Admin panel
│   └── parent/              # Parent dashboard
├── components/              # React components
├── lib/                     # Utilities
├── hooks/                   # Custom hooks
└── styles/                  # Global styles
```

## Features

- Landing page
- User dashboard (web access to mobile features)
- Admin panel for content management
- Parent dashboard for monitoring progress

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Styling**: Tailwind CSS
- **Auth**: Supabase Auth
- **Charts**: Recharts
- **Animations**: Framer Motion
