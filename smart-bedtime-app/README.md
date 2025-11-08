# 🌙 FlowBed - Smart Bedtime for Developers

A beautiful, intelligent bedtime reminder app designed specifically for developers who work in flow states. FlowBed integrates with your calendar and respects your deep focus time, never interrupting when you're in the zone.

## ✨ Features

- **Dual Bedtime Settings**: Set different bedtimes for work nights and free nights
- **Calendar Integration**: Automatically detects if you have tasks tomorrow
- **Flow State Detection**: Won't interrupt you when you're in deep focus
- **Smart Notifications**: Gentle browser notifications when it's time for bed
- **Beautiful UI**: Apple-inspired design with smooth animations and micro-interactions
- **Real-time Clock**: Always shows current time with elegant typography
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile

## 🎯 The Problem It Solves

As developers, we often lose track of time when we're in flow state. This can lead to:
- Staying up too late before important work days
- Inconsistent sleep schedules
- Burnout from overworking

FlowBed helps you maintain a healthy sleep schedule while respecting your creative process. It knows when you're in flow and won't interrupt those precious moments of deep work.

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ installed
- npm or yarn package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd smart-bedtime-app
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser and navigate to `http://localhost:5173`

### Building for Production

```bash
npm run build
```

The built files will be in the `dist` directory, ready to deploy to any static hosting service.

## 📖 How to Use

1. **Set Your Bedtimes**:
   - Configure your work night bedtime (when you have tasks tomorrow)
   - Configure your free night bedtime (when you have no tasks tomorrow)

2. **Connect Your Calendar** (simulated in current version):
   - Click "Connect" on the Calendar card
   - The app will check for events tomorrow and adjust your bedtime accordingly

3. **Manage Flow State**:
   - Click "Enter flow" when you're starting deep work
   - The app won't interrupt you with bedtime notifications while in flow
   - Remember to exit flow state when you're done!

4. **Enable Notifications**:
   - Allow browser notifications when prompted
   - You'll receive gentle reminders when bedtime arrives

## 🎨 Design Philosophy

FlowBed is built with obsessive attention to design detail:

- **Simplicity**: Clean, intuitive interface with no learning curve
- **Typography**: Beautiful font hierarchy with SF Pro-inspired system fonts
- **Spacing**: Generous whitespace for breathing room
- **Color**: Purposeful gradient palette (purple to violet)
- **Animations**: Smooth, delightful micro-interactions
- **Accessibility**: High contrast, readable text, keyboard navigation

Every pixel matters. Every interaction delights.

## 🔮 Future Enhancements

- **Real Google Calendar Integration**: OAuth2 flow for actual calendar access
- **Advanced Flow Detection**:
  - Git commit activity monitoring
  - IDE activity tracking
  - Keyboard/mouse pattern analysis
- **Sleep Pattern Analytics**: Track your sleep consistency over time
- **Smart Suggestions**: ML-based bedtime recommendations
- **Progressive Web App**: Install as native app on mobile/desktop
- **Custom Themes**: Dark mode, custom color schemes
- **Multiple Calendar Support**: Outlook, Apple Calendar, etc.
- **Gentle Escalation**: Gradual reminder intensity if you ignore initial alert
- **Do Not Disturb Integration**: Respect system DND settings

## 🛠️ Tech Stack

- **React 19**: Latest React with concurrent features
- **Vite**: Lightning-fast build tool and dev server
- **CSS3**: Custom animations and glassmorphism effects
- **Browser Notifications API**: Native notification support
- **LocalStorage**: Persistent settings (future enhancement)

## 📱 Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Opera 76+

Note: Notification API support required for bedtime alerts.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - feel free to use this project however you'd like!

## 💡 Inspiration

Built for developers, by developers. Inspired by the need to balance productivity with health.

> "The best way to predict the future is to invent it, but first, get enough sleep."

---

Made with 💜 and late nights (ironically)
