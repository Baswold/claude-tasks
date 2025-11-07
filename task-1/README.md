# AI Inference Calculator

A beautiful web application that helps you find the perfect hardware for running AI models. Built with stunning UI/UX, powered by HuggingFace's model registry and Claude AI for intelligent hardware recommendations.

## ✨ Features

- **Live Model Database**: Browse thousands of AI models from HuggingFace
- **Real-time Search**: Instantly filter models by name or organization
- **Detailed Specifications**: View parameters, RAM/VRAM requirements, and model types
- **AI-Powered Research**: Claude AI researches the best hardware for your selected model
- **Beautiful Design**: Modern, gradient-based UI with smooth animations and micro-interactions

## 🎯 Use Case Example

```
Model: Kimi K2 (or any LLM)
Parameters: 1 Trillion
RAM/VRAM: Calculated automatically
Result: Claude researches and recommends optimal hardware
```

## 🚀 Quick Start

### Prerequisites

- Node.js 16+ installed
- Anthropic API key ([Get one here](https://console.anthropic.com/))

### Installation

1. **Clone or navigate to this directory**

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_actual_api_key_here
   ```

4. **Start the server**
   ```bash
   npm start
   ```

   For development with auto-reload:
   ```bash
   npm run dev
   ```

5. **Open your browser**
   Navigate to `http://localhost:3000`

## 🎨 Design Philosophy

This application embodies the principle that **beautiful design is functional design**. Every pixel has been crafted with attention to detail:

- **Gradient Orbs**: Animated background creates depth and visual interest
- **Glass Morphism**: Semi-transparent cards with blur effects
- **Micro-interactions**: Hover states, loading animations, smooth transitions
- **Typography**: Inter font family for maximum readability
- **Responsive**: Works beautifully on desktop and mobile

## 🏗️ Architecture

```
task-1/
├── index.html          # Main HTML with semantic structure
├── styles.css          # Beautiful CSS with animations and gradients
├── app.js              # Frontend JavaScript for interactions
├── server.js           # Express backend with API endpoints
├── package.json        # Dependencies and scripts
├── .env.example        # Environment variable template
└── README.md          # This file
```

## 🔧 API Endpoints

- `GET /api/models` - Fetch available AI models from HuggingFace
- `GET /api/model-details/:modelId` - Get detailed specs for a model
- `POST /api/find-hardware` - Use Claude AI to research hardware recommendations

## 🎭 Technologies Used

- **Frontend**: Vanilla JavaScript, CSS3 with advanced animations
- **Backend**: Node.js, Express
- **AI**: Anthropic Claude API (Sonnet 4.5)
- **Data**: HuggingFace Model Hub API
- **Design**: Custom gradient system, glass morphism, Inter font

## 🌟 Key Features Explained

### Live Model Updates
The model list is fetched from HuggingFace and cached for 5 minutes to ensure you always have access to the latest models while maintaining performance.

### Automatic Memory Calculation
RAM and VRAM requirements are calculated based on:
- **FP16 Precision**: 2 bytes per parameter
- **RAM**: 1.2x model size (20% overhead for system operations)
- **VRAM**: 1.3x model size (30% for activations and gradients)

### Claude AI Research
When you click "Find Perfect Hardware", Claude uses its built-in web search to:
1. Research current GPU market
2. Find best consumer and enterprise options
3. Calculate approximate costs
4. Suggest cloud alternatives
5. Provide optimization tips

## 🛠️ Development

The codebase is designed to be easily extensible:

- Add new model sources in `server.js`
- Customize UI colors and gradients in `styles.css`
- Extend API endpoints for additional features
- Modify Claude's research prompt for different recommendations

## 📱 Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## 🤝 Contributing

This is a demonstration project showcasing beautiful UI/UX combined with AI capabilities. Feel free to fork and customize for your needs.

## 📄 License

MIT

---

**Built with ❤️ and an obsessive attention to detail**
