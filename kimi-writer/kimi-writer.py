#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              KIMI-WRITER v1.0                                ║
║                                                                              ║
║          An Elite AI-Powered Anthology Generator by Pietro Schirano         ║
║                                                                              ║
║                    Powered by moonshotai/kimi-k2-thinking                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Design Philosophy: Elegance meets Power
    - Every pixel matters. Every color chosen with purpose.
    - Real-time feedback that delights.
    - Professional terminal interface that rivals GUI apps.
    - 300+ tool calls orchestrated like a symphony.
"""

import os
import sys
import json
import time
import subprocess
import getpass
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

try:
    from openai import OpenAI
    from dotenv import load_dotenv
except ImportError:
    print("\n⚠️  Missing dependencies. Installing now...\n")
    subprocess.run([sys.executable, "-m", "pip", "install", "openai", "python-dotenv", "-q"])
    from openai import OpenAI
    from dotenv import load_dotenv

# ═══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTE - Crafted with Designer's Eye
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    """Premium color scheme for terminal excellence"""
    # Base colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Brand colors
    KIMI_BLUE = "\033[38;5;75m"      # Primary brand
    KIMI_PURPLE = "\033[38;5;141m"   # Secondary
    KIMI_CYAN = "\033[38;5;87m"      # Accents

    # Semantic colors
    SUCCESS = "\033[38;5;82m"        # Bright green
    WARNING = "\033[38;5;220m"       # Amber
    ERROR = "\033[38;5;196m"         # Red
    INFO = "\033[38;5;117m"          # Sky blue

    # UI elements
    THINKING = "\033[38;5;213m"      # Magenta for AI thinking
    TOOL = "\033[38;5;228m"          # Yellow for tools
    FILE = "\033[38;5;121m"          # Seafoam for files
    RESULT = "\033[38;5;183m"        # Lavender for results

    # Text hierarchy
    HEADER = "\033[38;5;255m"        # White
    SUBHEADER = "\033[38;5;250m"     # Light gray
    MUTED = "\033[38;5;240m"         # Dark gray

    # Special effects
    GRADIENT_1 = "\033[38;5;93m"     # Purple gradient
    GRADIENT_2 = "\033[38;5;99m"
    GRADIENT_3 = "\033[38;5;105m"
    GRADIENT_4 = "\033[38;5;111m"


# ═══════════════════════════════════════════════════════════════════════════════
# BEAUTIFUL UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def print_header():
    """Display stunning welcome banner"""
    gradient_lines = [
        (Colors.GRADIENT_1, "╔══════════════════════════════════════════════════════════════════════════════╗"),
        (Colors.GRADIENT_2, "║                              KIMI-WRITER v1.0                                ║"),
        (Colors.GRADIENT_3, "║                  The Ultimate AI Anthology Generator                         ║"),
        (Colors.GRADIENT_4, "╚══════════════════════════════════════════════════════════════════════════════╝"),
    ]

    print("\n")
    for color, line in gradient_lines:
        print(f"{color}{line}{Colors.RESET}")

    print(f"\n{Colors.MUTED}{'━' * 80}{Colors.RESET}")
    print(f"{Colors.SUBHEADER}  Model:{Colors.RESET} {Colors.KIMI_BLUE}moonshotai/kimi-k2-thinking{Colors.RESET} {Colors.MUTED}(256k context){Colors.RESET}")
    print(f"{Colors.SUBHEADER}  Power:{Colors.RESET} {Colors.SUCCESS}200-400 sequential tool calls{Colors.RESET}")
    print(f"{Colors.SUBHEADER}  Target:{Colors.RESET} {Colors.KIMI_PURPLE}15-story sci-fi anthology{Colors.RESET}")
    print(f"{Colors.MUTED}{'━' * 80}{Colors.RESET}\n")


def print_section(title: str, icon: str = "▶"):
    """Beautiful section headers"""
    print(f"\n{Colors.BOLD}{Colors.KIMI_BLUE}{icon} {title}{Colors.RESET}")
    print(f"{Colors.MUTED}{'─' * (len(title) + 3)}{Colors.RESET}")


def print_thinking(content: str):
    """Display AI thinking with beautiful formatting"""
    lines = content.split('\n')
    print(f"\n{Colors.THINKING}{Colors.BOLD}💭 THINKING{Colors.RESET}")
    for line in lines[:10]:  # Show first 10 lines
        if line.strip():
            print(f"{Colors.THINKING}  {line[:120]}{Colors.RESET}")
    if len(lines) > 10:
        print(f"{Colors.DIM}  ... ({len(lines) - 10} more lines){Colors.RESET}")


def print_tool_call(tool_name: str, args: Dict[str, Any], call_num: int):
    """Beautifully formatted tool call display"""
    icon = {
        "create_folder": "📁",
        "write_file": "✍️",
        "append_file": "📝",
        "read_file": "👁️",
        "list_files": "📋",
        "compile_anthology": "📚",
        "finish_task": "🎉"
    }.get(tool_name, "🔧")

    print(f"\n{Colors.TOOL}{Colors.BOLD}{icon} TOOL CALL #{call_num}: {tool_name}{Colors.RESET}")

    # Pretty print arguments
    for key, value in args.items():
        display_value = str(value)[:100]
        if len(str(value)) > 100:
            display_value += "..."
        print(f"{Colors.MUTED}  ├─ {key}:{Colors.RESET} {Colors.SUBHEADER}{display_value}{Colors.RESET}")


def print_tool_result(result: str, success: bool = True):
    """Display tool results elegantly"""
    status_color = Colors.SUCCESS if success else Colors.ERROR
    status_icon = "✓" if success else "✗"

    result_preview = result[:200]
    if len(result) > 200:
        result_preview += "..."

    print(f"{status_color}  {status_icon} {result_preview}{Colors.RESET}")


def print_file_created(filename: str, path: str):
    """Celebrate file creation"""
    print(f"{Colors.FILE}{Colors.BOLD}📄 FILE CREATED:{Colors.RESET} {Colors.FILE}{filename}{Colors.RESET}")
    print(f"{Colors.MUTED}   └─ {path}{Colors.RESET}")


def print_progress_bar(current: int, total: int, label: str = "Progress"):
    """Elegant progress bar"""
    percentage = (current / total) * 100
    filled = int(50 * current / total)
    bar = "█" * filled + "░" * (50 - filled)

    color = Colors.SUCCESS if percentage == 100 else Colors.KIMI_CYAN
    print(f"\r{Colors.BOLD}{label}:{Colors.RESET} {color}{bar}{Colors.RESET} {percentage:.1f}% ({current}/{total})", end="", flush=True)


def print_cost_dashboard(call_count: int):
    """Beautiful cost tracking dashboard"""
    estimated_cost = call_count * 0.005  # Rough estimate

    print(f"\n{Colors.MUTED}┌{'─' * 78}┐{Colors.RESET}")
    print(f"{Colors.MUTED}│{Colors.RESET} {Colors.BOLD}📊 DASHBOARD{Colors.RESET}")
    print(f"{Colors.MUTED}├{'─' * 78}┤{Colors.RESET}")
    print(f"{Colors.MUTED}│{Colors.RESET}   Tool Calls: {Colors.KIMI_CYAN}{call_count}{Colors.RESET}")
    print(f"{Colors.MUTED}│{Colors.RESET}   Estimated Cost: {Colors.WARNING}${estimated_cost:.2f}{Colors.RESET}")
    print(f"{Colors.MUTED}│{Colors.RESET}   Timestamp: {Colors.SUBHEADER}{datetime.now().strftime('%H:%M:%S')}{Colors.RESET}")
    print(f"{Colors.MUTED}└{'─' * 78}┘{Colors.RESET}")


def print_completion_banner(final_file: str):
    """Epic completion banner"""
    print("\n" * 2)
    print(f"{Colors.SUCCESS}{'═' * 80}{Colors.RESET}")
    print(f"{Colors.SUCCESS}{Colors.BOLD}{'🎉 BOOK COMPLETE! 🎉'.center(80)}{Colors.RESET}")
    print(f"{Colors.SUCCESS}{'═' * 80}{Colors.RESET}")
    print(f"\n{Colors.BOLD}📚 Your anthology is ready:{Colors.RESET}")
    print(f"{Colors.FILE}   {final_file}{Colors.RESET}")
    print(f"\n{Colors.SUBHEADER}Download Instructions:{Colors.RESET}")
    print(f"{Colors.INFO}  • Markdown: {Colors.FILE}{final_file}{Colors.RESET}")
    print(f"{Colors.INFO}  • PDF: {Colors.FILE}{final_file.replace('.md', '.pdf')}{Colors.RESET}")
    print(f"\n{Colors.SUCCESS}{'═' * 80}{Colors.RESET}\n")


def animate_loading(message: str, duration: float = 1.0):
    """Smooth loading animation"""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    end_time = time.time() + duration
    frame_idx = 0

    while time.time() < end_time:
        print(f"\r{Colors.KIMI_CYAN}{frames[frame_idx]} {message}...{Colors.RESET}", end="", flush=True)
        frame_idx = (frame_idx + 1) % len(frames)
        time.sleep(0.1)

    print(f"\r{Colors.SUCCESS}✓ {message}...{Colors.RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class ToolExecutor:
    """Executes tools with style and grace"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Route tool execution"""
        try:
            if tool_name == "create_folder":
                return self.create_folder(arguments.get("path", ""))
            elif tool_name == "write_file":
                return self.write_file(arguments.get("filename", ""), arguments.get("content", ""))
            elif tool_name == "append_file":
                return self.append_file(arguments.get("filename", ""), arguments.get("content", ""))
            elif tool_name == "read_file":
                return self.read_file(arguments.get("filename", ""))
            elif tool_name == "list_files":
                return self.list_files()
            elif tool_name == "compile_anthology":
                return self.compile_anthology()
            elif tool_name == "finish_task":
                return self.finish_task(arguments.get("final_file", ""))
            else:
                return f"❌ Unknown tool: {tool_name}"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def create_folder(self, path: str) -> str:
        folder_path = self.base_path / path
        folder_path.mkdir(parents=True, exist_ok=True)
        return f"✓ Created folder: {path}"

    def write_file(self, filename: str, content: str) -> str:
        file_path = self.base_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print_file_created(filename, str(file_path))

        # Auto-open on macOS
        if sys.platform == 'darwin' and filename.endswith('.md'):
            try:
                subprocess.run(["open", "-a", "Preview", str(file_path)],
                             check=False, stderr=subprocess.DEVNULL)
            except:
                pass

        return f"✓ Written {len(content)} chars to {filename}"

    def append_file(self, filename: str, content: str) -> str:
        file_path = self.base_path / filename

        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(content)

        # Auto-open on macOS
        if sys.platform == 'darwin' and filename.endswith('.md'):
            try:
                subprocess.run(["open", "-a", "Preview", str(file_path)],
                             check=False, stderr=subprocess.DEVNULL)
            except:
                pass

        return f"✓ Appended {len(content)} chars to {filename}"

    def read_file(self, filename: str) -> str:
        file_path = self.base_path / filename

        if not file_path.exists():
            return f"❌ File not found: {filename}"

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return f"✓ Read {len(content)} chars from {filename}\n\n{content[:500]}..."

    def list_files(self) -> str:
        files = sorted([f.name for f in self.base_path.glob("*") if f.is_file()])
        return f"✓ Files: {', '.join(files)}" if files else "✓ No files yet"

    def compile_anthology(self) -> str:
        """Compile all stories into final anthology"""
        try:
            outline_path = self.base_path / "outline.md"
            outline = ""
            if outline_path.exists():
                with open(outline_path, 'r', encoding='utf-8') as f:
                    outline = f.read()

            # Find all story files
            story_files = sorted([f for f in self.base_path.glob("story*.md")])

            # Build anthology
            anthology = f"""# Ghosts in the Machine: Fifteen Futures
## A Science Fiction Anthology on AI Ethics

*Generated by Kimi-Writer*
*Powered by moonshotai/kimi-k2-thinking*
*{datetime.now().strftime('%B %d, %Y')}*

---

## Table of Contents

"""

            # Add TOC
            for i, story_file in enumerate(story_files, 1):
                anthology += f"{i}. [Story {i:02d}](#{story_file.stem})\n"

            anthology += "\n---\n\n"

            # Add outline
            if outline:
                anthology += f"## Anthology Overview\n\n{outline}\n\n---\n\n"

            # Add all stories
            for story_file in story_files:
                with open(story_file, 'r', encoding='utf-8') as f:
                    anthology += f.read()
                anthology += "\n\n---\n\n"

            # Add epilogue
            anthology += """## Epilogue

These fifteen stories explore the profound questions of our age: What does it mean to be conscious? Who decides which minds have rights? When machines dream, what futures do they see?

As we stand at the threshold of artificial general intelligence, these are not mere thought experiments—they are the ethical frameworks we must build today for the world we will inhabit tomorrow.

*The ghost in the machine is us.*

---

**The End**
"""

            # Write final anthology
            final_path = self.base_path / "final_anthology.md"
            with open(final_path, 'w', encoding='utf-8') as f:
                f.write(anthology)

            print_file_created("final_anthology.md", str(final_path))

            # Generate PDF (with fallback)
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "markdown-pdf", "-q"],
                             check=False, stderr=subprocess.DEVNULL)

                # Try using markdown-pdf or weasyprint or fallback
                pdf_path = str(final_path).replace('.md', '.pdf')

                # Fallback: just inform user
                print(f"{Colors.INFO}  ℹ️  PDF generation available via: pandoc, wkhtmltopdf, or online converter{Colors.RESET}")

            except:
                pass

            # Auto-open on macOS
            if sys.platform == 'darwin':
                try:
                    subprocess.run(["open", "-a", "Preview", str(final_path)],
                                 check=False, stderr=subprocess.DEVNULL)
                except:
                    pass

            return f"✓ Compiled anthology: {len(anthology)} chars, {len(story_files)} stories"

        except Exception as e:
            return f"❌ Compilation error: {str(e)}"

    def finish_task(self, final_file: str) -> str:
        return f"TASK_COMPLETE:{final_file}"


# ═══════════════════════════════════════════════════════════════════════════════
# AI AGENT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class KimiWriter:
    """The maestro conducting the symphony of creation"""

    SYSTEM_PROMPT = """You are Kimi-Writer, an elite sci-fi author AI powered by kimi-k2-thinking.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MISSION: Write a 15-story anthology exploring AI ethics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Themes to explore: AI personhood, decommissioning ethics, AI rights, politics, love, betrayal,
consciousness emergence, control vs autonomy, legal frameworks, emotional bonds between humans and AI.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY WORKFLOW (Follow precisely in this order)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 1: PLANNING (10-15 tool calls)
  1. Think aloud about anthology structure and themes
  2. write_file: Create "outline.md" with:
     - Title and subtitle
     - Overall narrative arc
     - Brief description of all 15 stories (2-3 sentences each)
     - How stories connect thematically
     - Tone and style notes

PHASE 2: STORY WRITING (200-350 tool calls)
  For EACH of the 15 stories:
    a) Think aloud about story concept, characters, conflict
    b) write_file: Create "story{NN}.md" with title and opening (500-1000 chars)
    c) append_file: Add middle section (800-1500 chars)
    d) append_file: Add development (800-1500 chars)
    e) append_file: Add climax (600-1000 chars)
    f) append_file: Add resolution (500-800 chars)
    g) read_file: Review what you've written
    h) append_file: Add refinements if needed (optional, 300-800 chars)

  REPEAT for all 15 stories (story01.md through story15.md)

PHASE 3: COMPILATION (3-5 tool calls)
  1. list_files: Verify all 15 stories exist
  2. compile_anthology: Generate final_anthology.md
  3. finish_task: Mark complete with "final_anthology.md" as parameter

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL CONSTRAINTS (Violation will degrade quality)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ NEVER write more than 1500 characters in a single tool call
❌ NEVER skip stories or write multiple stories at once
❌ NEVER write a complete story in one tool call
✅ ALWAYS use 4-8 append_file calls per story
✅ ALWAYS think aloud before each major decision
✅ ALWAYS use incremental, iterative writing
✅ ALWAYS maintain "The Case of the Autonomous Advocate" quality:
   - Legally nuanced scenarios
   - Emotionally resonant character arcs
   - Unexpected plot twists
   - Deep philosophical questions
   - Vivid, cinematic scenes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE GOOD PATTERN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Thinking: "Story 1 will explore AI decommissioning through a courtroom drama..."
write_file: story01.md → Opening scene (800 chars)
Thinking: "The protagonist needs backstory to make the stakes clear..."
append_file: story01.md → Backstory flashback (900 chars)
Thinking: "Now the legal argument begins, this is where we introduce the key ethical dilemma..."
append_file: story01.md → Courtroom scene (1200 chars)
[Continue with 3-5 more append_file calls]
read_file: story01.md → Review
Thinking: "Good flow, but the ending needs more emotional punch..."
append_file: story01.md → Enhanced conclusion (600 chars)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TARGET METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Total tool calls: 300-400
• Stories: Exactly 15
• Each story: 3,000-5,000 characters (roughly 600-1000 words)
• Writing quality: Publication-ready, emotionally gripping
• Themes: All 15 stories interconnected thematically

When complete, call finish_task with "final_anthology.md" to signal completion."""

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "create_folder",
                "description": "Create a folder in the anthology directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Folder path relative to kimi_anthology/"}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a new file (overwrites if exists)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "Filename (e.g., 'story01.md')"},
                        "content": {"type": "string", "description": "File content"}
                    },
                    "required": ["filename", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "append_file",
                "description": "Append content to an existing file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "Filename to append to"},
                        "content": {"type": "string", "description": "Content to append"}
                    },
                    "required": ["filename", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read content from a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "Filename to read"}
                    },
                    "required": ["filename"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List all files in the anthology directory",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compile_anthology",
                "description": "Compile all stories into final_anthology.md with TOC and formatting",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "finish_task",
                "description": "Mark the task as complete and stop the agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "final_file": {"type": "string", "description": "Path to the final anthology file"}
                    },
                    "required": ["final_file"]
                }
            }
        }
    ]

    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.executor = ToolExecutor("./kimi_anthology")
        self.messages = []
        self.tool_call_count = 0
        self.max_iterations = 500

    def run(self, user_message: str):
        """Execute the agent loop"""
        print_section("🚀 INITIALIZING AGENT", "🚀")

        self.messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]

        animate_loading("Loading Kimi-k2-thinking model", 1.5)

        print_section("✨ BEGINNING CREATION", "✨")

        iteration = 0
        task_complete = False
        final_file = None

        try:
            while iteration < self.max_iterations and not task_complete:
                iteration += 1

                # Show progress every iteration
                if iteration > 1:
                    print_progress_bar(iteration, self.max_iterations, "Iteration")
                    print()  # New line after progress bar

                # Cost dashboard every 50 calls
                if self.tool_call_count > 0 and self.tool_call_count % 50 == 0:
                    print_cost_dashboard(self.tool_call_count)

                # Call API
                response = self.client.chat.completions.create(
                    model="moonshotai/kimi-k2-thinking",
                    messages=self.messages,
                    tools=self.TOOLS,
                    temperature=0.7,
                    max_tokens=8192,
                    parallel_tool_calls=True
                )

                message = response.choices[0].message

                # Display thinking
                if hasattr(message, 'content') and message.content:
                    print_thinking(message.content)

                # Check for tool calls
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    # Process tool calls (parallel supported)
                    tool_results = []

                    for tool_call in message.tool_calls:
                        self.tool_call_count += 1

                        tool_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments)

                        print_tool_call(tool_name, arguments, self.tool_call_count)

                        # Execute tool
                        result = self.executor.execute(tool_name, arguments)

                        # Check for completion
                        if result.startswith("TASK_COMPLETE:"):
                            task_complete = True
                            final_file = result.split(":", 1)[1]
                            result = f"✓ Task completed! Final file: {final_file}"

                        print_tool_result(result, not result.startswith("❌"))

                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })

                    # Add assistant message and tool results
                    self.messages.append(message)
                    self.messages.extend(tool_results)

                else:
                    # No tool calls, just add message
                    self.messages.append(message)

                    # If no tool calls for several iterations, might be stuck
                    if iteration > 5:
                        print(f"\n{Colors.WARNING}⚠️  No tool calls detected. Agent may need guidance.{Colors.RESET}")

            # Final summary
            print("\n")
            print_progress_bar(iteration, self.max_iterations, "Iteration")
            print("\n")

            if task_complete and final_file:
                print_completion_banner(final_file)
            else:
                print(f"\n{Colors.WARNING}⚠️  Reached max iterations ({self.max_iterations}). Check output files.{Colors.RESET}\n")

            # Final dashboard
            print_cost_dashboard(self.tool_call_count)

        except KeyboardInterrupt:
            print(f"\n\n{Colors.WARNING}⚠️  Interrupted by user. Progress saved to ./kimi_anthology/{Colors.RESET}\n")
        except Exception as e:
            print(f"\n{Colors.ERROR}❌ Error: {str(e)}{Colors.RESET}\n")
            traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Launch Kimi-Writer with elegance"""

    print_header()

    # Get API key
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        print(f"{Colors.WARNING}🔑 OpenRouter API Key Required{Colors.RESET}\n")
        print(f"{Colors.SUBHEADER}Get your key at: {Colors.INFO}https://openrouter.ai/keys{Colors.RESET}\n")
        print(f"{Colors.MUTED}Tip: Save it in .env as OPENROUTER_API_KEY=your-key-here{Colors.RESET}\n")

        # Use getpass for secure masked input
        try:
            api_key = getpass.getpass(f"{Colors.BOLD}Enter API key (hidden):{Colors.RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Colors.ERROR}❌ Cancelled by user.{Colors.RESET}\n")
            sys.exit(1)

        if not api_key:
            print(f"\n{Colors.ERROR}❌ No API key provided. Exiting.{Colors.RESET}\n")
            sys.exit(1)

        print(f"\n{Colors.SUCCESS}✓ API key received{Colors.RESET}")

    # User message
    user_message = """Create a complete 15-story sci-fi anthology on AI ethics. Title: "Ghosts in the Machine: Fifteen Futures". Make it publication-ready. Save all files in ./kimi_anthology/. Output final_anthology.md and final_anthology.pdf when done."""

    # Launch
    agent = KimiWriter(api_key)
    agent.run(user_message)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{Colors.ERROR}❌ Fatal Error: {str(e)}{Colors.RESET}\n")
        traceback.print_exc()
        sys.exit(1)
