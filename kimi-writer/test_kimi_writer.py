#!/usr/bin/env python3
"""
Comprehensive Test Suite for Kimi-Writer
Tests UI components, tool execution, and system integrity
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path and import the module dynamically
sys.path.insert(0, str(Path(__file__).parent))

# Import dynamically since filename has hyphen
import importlib.util
spec = importlib.util.spec_from_file_location("kimi_writer", Path(__file__).parent / "kimi-writer.py")
kimi_writer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kimi_writer)

# Import components to test
Colors = kimi_writer.Colors
ToolExecutor = kimi_writer.ToolExecutor
print_header = kimi_writer.print_header
print_section = kimi_writer.print_section
print_thinking = kimi_writer.print_thinking
print_tool_call = kimi_writer.print_tool_call
print_tool_result = kimi_writer.print_tool_result
print_file_created = kimi_writer.print_file_created
print_progress_bar = kimi_writer.print_progress_bar
print_cost_dashboard = kimi_writer.print_cost_dashboard
print_completion_banner = kimi_writer.print_completion_banner

def test_ui_components():
    """Test all UI components render beautifully"""
    print(f"\n{Colors.BOLD}{Colors.KIMI_BLUE}═══ TEST 1: UI COMPONENTS ═══{Colors.RESET}\n")

    # Test header
    print_header()

    # Test sections
    print_section("Testing Section Headers", "✨")

    # Test thinking display
    print_thinking("""I'm analyzing the story structure...
The protagonist needs depth.
Let me craft the opening scene with care.
This will resonate emotionally.
Building tension through dialogue.
The twist should come in act three.
Philosophical questions about consciousness.
Legal framework for AI rights.
Character development is key.
Plot threads need to converge.
Extra line 1
Extra line 2""")

    # Test tool calls
    print_tool_call("write_file", {"filename": "test.md", "content": "Hello world"}, 42)
    print_tool_result("✓ Written 11 chars to test.md", True)

    print_tool_call("append_file", {"filename": "story01.md", "content": "Chapter continues..."}, 43)
    print_tool_result("✓ Appended 19 chars to story01.md", True)

    # Test file creation
    print_file_created("story01.md", "/path/to/kimi_anthology/story01.md")

    # Test progress bar
    for i in range(0, 101, 20):
        print_progress_bar(i, 100, "Story Generation")
        import time
        time.sleep(0.3)
    print()

    # Test cost dashboard
    print_cost_dashboard(156)

    # Test completion banner
    print_completion_banner("./kimi_anthology/final_anthology.md")

    print(f"\n{Colors.SUCCESS}✓ All UI components rendered successfully!{Colors.RESET}\n")


def test_tool_executor():
    """Test tool execution in isolated environment"""
    print(f"\n{Colors.BOLD}{Colors.KIMI_BLUE}═══ TEST 2: TOOL EXECUTOR ═══{Colors.RESET}\n")

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="kimi_test_")
    print(f"{Colors.INFO}📁 Test directory: {temp_dir}{Colors.RESET}\n")

    try:
        executor = ToolExecutor(temp_dir)

        # Test 1: Create folder
        print(f"{Colors.SUBHEADER}Test 1: Create Folder{Colors.RESET}")
        result = executor.execute("create_folder", {"path": "drafts"})
        print(f"  Result: {result}")
        assert "✓" in result, "Folder creation failed"
        assert Path(temp_dir, "drafts").exists(), "Folder not found"
        print(f"{Colors.SUCCESS}  ✓ PASS{Colors.RESET}\n")

        # Test 2: Write file
        print(f"{Colors.SUBHEADER}Test 2: Write File{Colors.RESET}")
        result = executor.execute("write_file", {
            "filename": "test.md",
            "content": "# Test Story\n\nThis is a test."
        })
        print(f"  Result: {result}")
        assert "✓" in result, "Write failed"
        assert Path(temp_dir, "test.md").exists(), "File not found"
        print(f"{Colors.SUCCESS}  ✓ PASS{Colors.RESET}\n")

        # Test 3: Read file
        print(f"{Colors.SUBHEADER}Test 3: Read File{Colors.RESET}")
        result = executor.execute("read_file", {"filename": "test.md"})
        print(f"  Result: {result[:80]}...")
        assert "Test Story" in result, "Content mismatch"
        print(f"{Colors.SUCCESS}  ✓ PASS{Colors.RESET}\n")

        # Test 4: Append file
        print(f"{Colors.SUBHEADER}Test 4: Append File{Colors.RESET}")
        result = executor.execute("append_file", {
            "filename": "test.md",
            "content": "\n\n## Chapter 2\n\nMore content."
        })
        print(f"  Result: {result}")
        content = Path(temp_dir, "test.md").read_text()
        assert "Chapter 2" in content, "Append failed"
        print(f"{Colors.SUCCESS}  ✓ PASS{Colors.RESET}\n")

        # Test 5: List files
        print(f"{Colors.SUBHEADER}Test 5: List Files{Colors.RESET}")
        result = executor.execute("list_files", {})
        print(f"  Result: {result}")
        assert "test.md" in result, "File not listed"
        print(f"{Colors.SUCCESS}  ✓ PASS{Colors.RESET}\n")

        # Test 6: Write multiple story files for compilation
        print(f"{Colors.SUBHEADER}Test 6: Write Story Files{Colors.RESET}")
        executor.execute("write_file", {
            "filename": "outline.md",
            "content": "# Anthology Outline\n\n15 stories about AI ethics."
        })
        for i in range(1, 4):
            executor.execute("write_file", {
                "filename": f"story{i:02d}.md",
                "content": f"# Story {i}\n\nThis is story number {i}."
            })
        print(f"{Colors.SUCCESS}  ✓ Created outline and 3 stories{Colors.RESET}\n")

        # Test 7: Compile anthology
        print(f"{Colors.SUBHEADER}Test 7: Compile Anthology{Colors.RESET}")
        result = executor.execute("compile_anthology", {})
        print(f"  Result: {result}")
        assert "✓" in result, "Compilation failed"
        assert Path(temp_dir, "final_anthology.md").exists(), "Anthology not created"

        # Verify content
        anthology = Path(temp_dir, "final_anthology.md").read_text()
        assert "Ghosts in the Machine" in anthology, "Title missing"
        assert "Table of Contents" in anthology, "TOC missing"
        assert "Story 1" in anthology, "Story missing"
        print(f"{Colors.SUCCESS}  ✓ PASS{Colors.RESET}\n")

        # Test 8: Finish task
        print(f"{Colors.SUBHEADER}Test 8: Finish Task{Colors.RESET}")
        result = executor.execute("finish_task", {"final_file": "final_anthology.md"})
        print(f"  Result: {result}")
        assert "TASK_COMPLETE" in result, "Finish task failed"
        print(f"{Colors.SUCCESS}  ✓ PASS{Colors.RESET}\n")

        print(f"{Colors.SUCCESS}✓ All tool executor tests passed!{Colors.RESET}\n")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"{Colors.MUTED}🧹 Cleaned up test directory{Colors.RESET}\n")


def test_syntax():
    """Verify Python syntax is valid"""
    print(f"\n{Colors.BOLD}{Colors.KIMI_BLUE}═══ TEST 3: SYNTAX VALIDATION ═══{Colors.RESET}\n")

    script_path = Path(__file__).parent / "kimi-writer.py"

    import py_compile
    try:
        py_compile.compile(str(script_path), doraise=True)
        print(f"{Colors.SUCCESS}✓ Python syntax is valid!{Colors.RESET}\n")
    except py_compile.PyCompileError as e:
        print(f"{Colors.ERROR}✗ Syntax error: {e}{Colors.RESET}\n")
        sys.exit(1)


def test_imports():
    """Verify all imports work"""
    print(f"\n{Colors.BOLD}{Colors.KIMI_BLUE}═══ TEST 4: IMPORT VALIDATION ═══{Colors.RESET}\n")

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("kimi_writer", Path(__file__).parent / "kimi-writer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"{Colors.SUCCESS}✓ Module imports successfully!{Colors.RESET}\n")
    except Exception as e:
        print(f"{Colors.ERROR}✗ Import error: {e}{Colors.RESET}\n")
        sys.exit(1)


def run_all_tests():
    """Execute complete test suite"""
    print(f"\n{Colors.BOLD}{Colors.KIMI_PURPLE}{'═' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.KIMI_PURPLE}{'KIMI-WRITER TEST SUITE'.center(80)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.KIMI_PURPLE}{'═' * 80}{Colors.RESET}\n")

    try:
        test_syntax()
        test_imports()
        test_ui_components()
        test_tool_executor()

        print(f"\n{Colors.SUCCESS}{Colors.BOLD}{'═' * 80}{Colors.RESET}")
        print(f"{Colors.SUCCESS}{Colors.BOLD}{'🎉 ALL TESTS PASSED! 🎉'.center(80)}{Colors.RESET}")
        print(f"{Colors.SUCCESS}{Colors.BOLD}{'═' * 80}{Colors.RESET}\n")

        return True

    except Exception as e:
        print(f"\n{Colors.ERROR}{Colors.BOLD}{'═' * 80}{Colors.RESET}")
        print(f"{Colors.ERROR}{Colors.BOLD}{'❌ TESTS FAILED ❌'.center(80)}{Colors.RESET}")
        print(f"{Colors.ERROR}{Colors.BOLD}{'═' * 80}{Colors.RESET}")
        print(f"\n{Colors.ERROR}Error: {str(e)}{Colors.RESET}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
