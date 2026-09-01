"""Terminal User Interface (TUI) full-pane layout for cn watch."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Any, Callable

from .cli import format_output_links


class WatcherTUI:
    """Full-pane terminal application managing output viewport, prompt input, and bottom status bar."""

    def __init__(
        self,
        folder: Path,
        model_name: str,
        on_submit: Callable[[str], None],
        on_reset: Callable[[], None],
        on_status: Callable[[], str],
        on_exit: Callable[[], None],
    ):
        self.folder = folder
        self.model_name = model_name
        self.on_submit = on_submit
        self.on_reset = on_reset
        self.on_status = on_status
        self.on_exit = on_exit

        self.tokens_total = 0
        self.tokens_prompt = 0
        self.tokens_completion = 0
        self.turn_count = 0
        self.last_tool_count = 0
        self.running = False
        # Transcript rendering calls ``render_bottom_chrome`` while already
        # holding this lock.  A re-entrant lock keeps rendering atomic while
        # allowing that nested call (the first startup transcript otherwise
        # deadlocks the TUI before the prompt is shown).
        self.lock = threading.RLock()

    def enter_screen(self):
        """Switch to alternate screen buffer and configure scrolling margins."""
        # Enter alternate screen buffer: \033[?1049h
        # Clear screen: \033[2J
        # Move cursor to 1,1: \033[H
        sys.stdout.write("\033[?1049h\033[2J\033[H")
        sys.stdout.flush()
        self.update_layout()

    def exit_screen(self):
        """Restore normal screen buffer."""
        # Reset scroll margins: \033[r
        # Exit alternate screen: \033[?1049l
        # Show cursor: \033[?25h
        sys.stdout.write("\033[r\033[?1049l\033[?25h\n")
        sys.stdout.flush()

    def get_dimensions(self) -> tuple[int, int]:
        """Get current terminal width and height."""
        size = shutil.get_terminal_size((80, 24))
        return size.columns, max(10, size.lines)

    def update_layout(self):
        """Set terminal scroll region for rows 1 to H-3, reserving H-2 (border), H-1 (prompt), and H (status bar)."""
        cols, rows = self.get_dimensions()
        scroll_bottom = max(1, rows - 3)
        # Set scrolling region to lines 1..scroll_bottom: \033[<top>;<bottom>r
        sys.stdout.write(f"\033[1;{scroll_bottom}r")
        sys.stdout.flush()
        self.render_bottom_chrome()

    def render_bottom_chrome(self, prompt_text: str = ""):
        """Render fixed divider border, prompt line, and bottom status bar."""
        with self.lock:
            cols, rows = self.get_dimensions()
            border_row = rows - 2
            prompt_row = rows - 1
            status_row = rows

            # 1. Divider Border (row H-2)
            sys.stdout.write(f"\033[{border_row};1H\033[2K\033[36m{'─' * cols}\033[0m")

            # 2. Prompt Line (row H-1)
            short_folder = self.folder.name or str(self.folder)
            prompt_prefix = f"\033[36m[{short_folder}]\033[0m ❯ "
            sys.stdout.write(f"\033[{prompt_row};1H\033[2K{prompt_prefix}{prompt_text}")

            # 3. Status Bar (row H) - Inverse / highlighted banner
            short_model = self.model_name.split("/")[-1]
            token_info = (
                f"📊 {self.tokens_total:,} tokens (turn {self.turn_count})"
                if self.tokens_total > 0
                else "📊 0 tokens"
            )
            tool_info = f" | {self.last_tool_count} tools" if self.last_tool_count > 0 else ""
            status_text = f" 📁 {short_folder}  │  🧠 {short_model}  │  {token_info}{tool_info}  │  /help /reset /exit "
            
            # Truncate or pad status text to fill the full width
            if len(status_text) > cols:
                status_text = status_text[:cols]
            else:
                status_text = status_text + " " * (cols - len(status_text))

            # Invert colors: \033[7m ... \033[0m
            sys.stdout.write(f"\033[{status_row};1H\033[2K\033[7m{status_text}\033[0m")

            # Return cursor to active prompt position
            cursor_col = len(f"[{short_folder}] ❯ ") + len(prompt_text) + 1
            sys.stdout.write(f"\033[{prompt_row};{cursor_col}H")
            sys.stdout.flush()

    def write_transcript(self, text: str, auto_scroll: bool = True):
        """Append text into the scrollable output viewport above the prompt."""
        with self.lock:
            cols, rows = self.get_dimensions()
            scroll_bottom = max(1, rows - 3)

            # Move cursor to bottom of the scrolling viewport
            sys.stdout.write(f"\033[{scroll_bottom};1H")
            
            # Format markdown & links cleanly
            formatted = format_output_links(text, mode="auto", wrap=True, theme="auto")
            for line in formatted.splitlines():
                sys.stdout.write(f"\n\033[2K{line}")
            
            sys.stdout.flush()
            self.render_bottom_chrome()

    def update_stats(self, total: int, prompt: int, completion: int, tool_count: int = 0):
        """Update live token and session counters."""
        self.tokens_total = total
        self.tokens_prompt = prompt
        self.tokens_completion = completion
        self.turn_count += 1
        self.last_tool_count = tool_count
        self.render_bottom_chrome()

    def run_loop(self, initial_logs: list[str] | None = None):
        """Main REPL loop for interactive TUI session."""
        self.running = True
        self.enter_screen()

        # Print startup logs after entering the alternate screen so they are
        # not erased by the screen initialization.
        if initial_logs:
            for log_line in initial_logs:
                self.write_transcript(log_line)

        # Print initial greeting in the transcript area
        self.write_transcript("🧭 Codebase-Navigator Interactive Console")
        self.write_transcript("   Type your question to ask about this codebase.")
        self.write_transcript("   Commands: /reset (clear context), /status (index info), /exit (quit)\n")

        while self.running:
            try:
                self.render_bottom_chrome()
                # Position cursor at prompt line
                cols, rows = self.get_dimensions()
                prompt_row = rows - 1
                short_folder = self.folder.name or str(self.folder)
                cursor_col = len(f"[{short_folder}] ❯ ") + 1
                sys.stdout.write(f"\033[{prompt_row};{cursor_col}H")
                sys.stdout.flush()

                # Read line
                line = sys.stdin.readline()
                if not line:  # EOF
                    break
                
                query = line.strip()
                if not query:
                    continue

                if query.lower() in ("/exit", "exit", "quit", ":q"):
                    break

                if query.lower() in ("/help", "help", "?"):
                    self.write_transcript("\n📖 Commands:")
                    self.write_transcript("  /reset  - Clear multi-turn conversation memory and restart fresh")
                    self.write_transcript("  /status - Show file count, chunk statistics, and active model")
                    self.write_transcript("  /exit   - Quit cn watch and restore terminal\n")
                    continue

                if query.lower() in ("/reset", "reset", "/clear", "clear"):
                    self.on_reset()
                    self.tokens_total = 0
                    self.tokens_prompt = 0
                    self.tokens_completion = 0
                    self.turn_count = 0
                    self.last_tool_count = 0
                    self.write_transcript("\n🧹 Conversation context reset. Started fresh session.\n")
                    self.render_bottom_chrome()
                    continue

                if query.lower() in ("/status", "status"):
                    status_text = self.on_status()
                    self.write_transcript(f"\n{status_text}\n")
                    continue

                # Echo user question in transcript
                self.write_transcript(f"\n\033[1;36m👤 You:\033[0m {query}\n")
                self.on_submit(query)

            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                self.write_transcript(f"\n❌ Error: {e}\n")

        self.exit_screen()
        self.on_exit()
