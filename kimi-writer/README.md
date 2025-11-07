# 🎭 Kimi-Writer

> **The Ultimate AI-Powered Anthology Generator**
> Create publication-ready 15-story sci-fi anthologies with 300+ intelligent tool calls

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-API-green.svg)](https://openrouter.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features

- 🤖 **Powered by Kimi-k2-thinking** - State-of-the-art agentic reasoning with 256k context
- 📚 **Complete Anthologies** - Generates 15 interconnected stories in any genre
- 🎨 **Beautiful Terminal UI** - Premium color scheme and real-time progress visualization
- 🎯 **Three Creation Modes** - Quick start, interactive chat, or detailed world-building
- 🔧 **300+ Tool Calls** - Incremental, iterative writing for maximum quality
- 📝 **Auto-Preview** - Opens Markdown files in Preview.app (macOS) after each write
- 💾 **Smart Compilation** - Automatic TOC generation and anthology formatting
- 🔒 **Secure Input** - Masked API key entry for security

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install openai python-dotenv
```

### 2. Get API Key

Sign up at [OpenRouter](https://openrouter.ai/keys) to get your API key.

### 3. Run Kimi-Writer

```bash
python kimi-writer.py
```

**Option A:** Enter API key when prompted (masked with getpass)
**Option B:** Create a `.env` file with:

```env
OPENROUTER_API_KEY=your-key-here
```

---

## 🎯 Three Creation Modes

Choose how you want to create your anthology:

### 1. 🚀 Quick Start
Perfect for: Getting started fast with minimal input

Simply provide a prompt (or use the default AI ethics anthology) and let Kimi-Writer handle everything.

```
Your prompt: "A cyberpunk anthology about memory trading"
```

### 2. 💬 Interactive Chat
Perfect for: Exploring ideas collaboratively

Have a conversation with Kimi about your vision. It will ask thoughtful questions about:
- Themes and genres you want to explore
- Target audience and tone
- Story structure preferences
- Character archetypes
- Key messages

After 3-5 exchanges, Kimi generates a custom plan based on your conversation.

### 3. 🌍 World-Builder
Perfect for: Authors with detailed worlds already planned

Provide comprehensive details through an interactive form:
- **World Building**: Setting name, time period, technology, society, locations, unique rules
- **Character Archetypes**: Protagonist types, antagonists, supporting cast
- **Themes & Tone**: Core themes, overall tone, message/takeaway
- **Story Preferences**: Writing style, how stories interconnect

Kimi will use your world-building sheet to craft 15 stories set in your universe.

---

## 📖 How It Works

Kimi-Writer follows a structured 3-phase workflow:

### Phase 1: Planning (10-15 tool calls)
- Analyzes themes and story structure
- Creates detailed `outline.md` with 15 story summaries
- Establishes narrative arc and tone

### Phase 2: Story Writing (200-350 tool calls)
For each of the 15 stories:
1. Thinks aloud about concept, characters, conflict
2. Creates initial file with opening scene
3. Iteratively appends 4-8 chunks per story (800-1500 chars each)
4. Reviews via `read_file` and refines
5. Ensures publication-ready quality

### Phase 3: Compilation (3-5 tool calls)
- Verifies all 15 stories exist
- Compiles into `final_anthology.md` with:
  - Title page
  - Table of contents
  - All 15 stories
  - Epilogue
- Marks task complete

---

## 🎨 Output Structure

```
./kimi_anthology/
├── outline.md              # Anthology overview and story summaries
├── story01.md             # Individual stories (15 total)
├── story02.md
├── ...
├── story15.md
└── final_anthology.md     # Complete compiled anthology
```

---

## 🎯 Quality Standards

Kimi-Writer maintains **"The Case of the Autonomous Advocate"** quality level:

- ⚖️ **Legally nuanced** scenarios exploring AI rights frameworks
- 💔 **Emotionally resonant** character arcs and relationships
- 🌀 **Unexpected plot twists** that challenge assumptions
- 🧠 **Deep philosophical questions** about consciousness and personhood
- 🎬 **Vivid, cinematic scenes** that immerse readers

---

## 🛠 Testing

Run the comprehensive test suite:

```bash
python test_kimi_writer.py
```

Tests cover:
- ✅ Python syntax validation
- ✅ Module imports
- ✅ UI component rendering
- ✅ Tool executor functionality
- ✅ File operations
- ✅ Anthology compilation

---

## 📊 Metrics

| Metric | Target |
|--------|--------|
| Total Tool Calls | 300-400 |
| Stories | Exactly 15 |
| Story Length | 3,000-5,000 chars each |
| Writing Quality | Publication-ready |
| Themes | Interconnected across anthology |

---

## 🎨 Design Philosophy

> "Every pixel matters. Every color chosen with purpose."

Kimi-Writer embodies Steve Jobs' design principles:

- **Elegance meets Power** - Beautiful terminal UI that rivals GUI apps
- **Real-time Feedback** - Progress bars, cost dashboards, live updates
- **Professional Polish** - Premium color palette, gradient banners
- **Attention to Detail** - Thoughtful spacing, icons, and formatting

---

## 🧠 System Prompt

Kimi-Writer uses a highly structured prompt with:

- **Mandatory Workflow** - 3 phases with clear steps
- **Critical Constraints** - Max 1500 chars per tool call, incremental writing
- **Example Patterns** - Shows good vs bad writing approaches
- **Target Metrics** - Clear quality and quantity goals

This ensures the AI follows instructions precisely and produces consistent, high-quality output.

---

## 📝 Example Run

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                              KIMI-WRITER v1.0                                ║
║                  The Ultimate AI Anthology Generator                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model: moonshotai/kimi-k2-thinking (256k context)
  Power: 200-400 sequential tool calls
  Target: 15-story sci-fi anthology
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 ▶ INITIALIZING AGENT
────────────────────────
✓ Loading Kimi-k2-thinking model...

✨ ▶ BEGINNING CREATION
───────────────────────

💭 THINKING
  I'm planning the anthology structure...
  15 stories exploring AI ethics from different angles...

✍️ TOOL CALL #1: write_file
  ├─ filename: outline.md
  ├─ content: # Ghosts in the Machine...
✓ Written 2847 chars to outline.md

[... 300+ more tool calls ...]

════════════════════════════════════════════════════════════════════════════════
                               🎉 BOOK COMPLETE! 🎉
════════════════════════════════════════════════════════════════════════════════

📚 Your anthology is ready:
   ./kimi_anthology/final_anthology.md

Download Instructions:
  • Markdown: ./kimi_anthology/final_anthology.md
  • PDF: ./kimi_anthology/final_anthology.pdf

════════════════════════════════════════════════════════════════════════════════
```

---

## 🎬 Inspired By

This project replicates [Pietro Schirano](https://twitter.com/skirano)'s viral Kimi-writer demo, showcasing the power of:

- **moonshotai/kimi-k2-thinking** - Native support for 200-400 sequential tool calls
- **Agentic Workflows** - Real-time thinking, planning, and incremental creation
- **Beautiful UX** - Terminal interfaces that delight and inform

---

## 📜 License

MIT License - Feel free to use, modify, and distribute!

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional compilation formats (EPUB, PDF via Pandoc)
- Custom theme support
- Interactive story selection
- Multi-language support
- Web UI version

---

## 💡 Tips

1. **Cost Management** - Monitor the cost dashboard every 50 calls
2. **API Keys** - Store in `.env` for convenience
3. **Quality First** - Let the AI take its time for best results
4. **macOS Auto-Preview** - Files open automatically in Preview.app
5. **Interruption** - Press Ctrl+C to stop gracefully (progress saved)

---

## 🌟 Star Us!

If you love Kimi-Writer, give us a star! ⭐

Built with ❤️ and 🤖 by the Kimi-Writer team.

---

**Ready to create your anthology? Run `python kimi-writer.py` and let the magic begin!**
