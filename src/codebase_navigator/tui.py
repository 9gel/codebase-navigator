"""Terminal User Interface (TUI) full-pane layout for cn watch."""

from __future__ import annotations

from pathlib import Path
import select
import shutil
import sys
import threading
import time
from typing import Any, Callable

from .cli import detect_terminal_theme, format_output_links


class WatcherTUI:
    """Full-pane terminal application managing output viewport, prompt input, and bottom status bar."""

    SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

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
        self.transcript_lines: list[str] = []
        self.scroll_offset = 0
        self.prompt_history: list[str] = []
        self.history_index = 0
        self.exit_notice = ""
        self.exit_notice_until = 0.0
        self.exit_notice_key = ""
        self.spinner_active = False
        self.spinner_frame = 0
        self._spinner_stop = threading.Event()
        self._spinner_thread: threading.Thread | None = None

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
        """Set the scroll region and reserve the fixed five-row bottom chrome."""
        cols, rows = self.get_dimensions()
        scroll_bottom = max(1, rows - 5)
        # Set scrolling region to lines 1..scroll_bottom: \033[<top>;<bottom>r
        sys.stdout.write(f"\033[1;{scroll_bottom}r")
        sys.stdout.flush()
        self.render_bottom_chrome()

    def render_bottom_chrome(self, prompt_text: str = ""):
        """Render fixed divider border, prompt line, and bottom status bar."""
        with self.lock:
            cols, rows = self.get_dimensions()
            border_row = rows - 4
            prompt_row = rows - 3
            blank_row = rows - 2
            status_row = rows - 1
            footer_row = rows

            # 1. Divider Border (row H-4)
            sys.stdout.write(f"\033[{border_row};1H\033[2K\033[36m{'─' * cols}\033[0m")

            # 2. Prompt Line (row H-3), with a blank separator before status.
            prompt_display = f" {prompt_text}"
            sys.stdout.write(f"\033[{prompt_row};1H\033[2K{prompt_display}")
            sys.stdout.write(f"\033[{blank_row};1H\033[2K")

            # 3. Status Bar (row H-1) - Inverse / highlighted banner
            short_folder = self.folder.name or str(self.folder)
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

            # 4. Static footer (row H)
            if self.exit_notice and time.monotonic() >= self.exit_notice_until:
                self.exit_notice = ""
                self.exit_notice_key = ""
            footer_text = self.exit_notice or " Keep `cn watch` up to ensure `cn search` work efficiently elsewhere."
            if len(footer_text) > cols:
                footer_text = footer_text[:cols]
            else:
                footer_text = footer_text + " " * (cols - len(footer_text))
            sys.stdout.write(f"\033[{footer_row};1H\033[2K{footer_text}")

            # Return cursor to active prompt position
            cursor_col = len(prompt_display) + 1
            sys.stdout.write(f"\033[{prompt_row};{cursor_col}H")
            sys.stdout.flush()

    def write_transcript(self, text: str, auto_scroll: bool = True):
        """Append text into the scrollable output viewport above the prompt."""
        with self.lock:
            # Format markdown & links cleanly
            formatted = format_output_links(text, mode="auto", wrap=True, theme="auto")
            lines = formatted.splitlines()
            self.transcript_lines.extend(lines or [""])
            if auto_scroll:
                self.scroll_offset = 0
            self.render()

    def start_spinner(self):
        """Show an animated transient line below the latest transcript output."""
        with self.lock:
            if self.spinner_active:
                return
            self.spinner_active = True
            self.spinner_frame = 0
            self._spinner_stop.clear()
            self.render()

        def animate():
            while not self._spinner_stop.wait(0.12):
                with self.lock:
                    if not self.spinner_active:
                        return
                    self.spinner_frame = (self.spinner_frame + 1) % len(self.SPINNER_FRAMES)
                    self.render()

        self._spinner_thread = threading.Thread(target=animate, daemon=True)
        self._spinner_thread.start()

    def stop_spinner(self):
        """Remove the transient spinner line and restore the full transcript viewport."""
        with self.lock:
            if not self.spinner_active:
                return
            self.spinner_active = False
            self._spinner_stop.set()
            thread = self._spinner_thread
            self._spinner_thread = None
            self.render()
        if thread and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _viewport_height(self) -> int:
        return max(1, self.get_dimensions()[1] - 5)

    def render_transcript(self):
        """Render the retained transcript according to the current scroll offset."""
        _, rows = self.get_dimensions()
        base_height = self._viewport_height()
        viewport_height = max(1, base_height - int(self.spinner_active))
        max_offset = max(0, len(self.transcript_lines) - viewport_height)
        self.scroll_offset = min(max(self.scroll_offset, 0), max_offset)
        end = len(self.transcript_lines) - self.scroll_offset
        start = max(0, end - viewport_height)

        for row in range(1, rows - 4):
            index = start + row - 1
            line = self.transcript_lines[index] if index < end else ""
            sys.stdout.write(f"\033[{row};1H\033[2K{line}")

        if self.spinner_active:
            spinner_row = viewport_height + 1
            frame = self.SPINNER_FRAMES[self.spinner_frame]
            sys.stdout.write(f"\033[{spinner_row};1H\033[2K {frame} Agent is working...")

    def render(self, prompt_text: str = ""):
        """Render the transcript and fixed terminal chrome."""
        with self.lock:
            self.render_transcript()
            self.render_bottom_chrome(prompt_text)

    def scroll_transcript(self, lines: int):
        """Scroll the transcript up (positive) or down (negative) by lines."""
        with self.lock:
            self.scroll_offset += lines
            self.render()

    def _read_key(self) -> tuple[str, str]:
        """Read one key or terminal escape sequence in cbreak mode."""
        if not select.select([sys.stdin], [], [], 0.2)[0]:
            return "timeout", ""
        first = sys.stdin.read(1)
        if not first:
            return "eof", ""
        if first != "\033":
            return "char", first

        sequence = first
        # Escape sequences arrive as a short burst. Avoid blocking forever on
        # a standalone Escape key while allowing mouse reports to complete.
        while select.select([sys.stdin], [], [], 0.05)[0]:
            sequence += sys.stdin.read(1)
            if sequence[-1] in "~ABCDMm":
                break

        if sequence in ("\033[A", "\033[B"):
            return ("up", "") if sequence.endswith("A") else ("down", "")
        if sequence == "\033[5~":
            return "page_up", ""
        if sequence == "\033[6~":
            return "page_down", ""
        if sequence.startswith("\033[<") and sequence.endswith(("M", "m")):
            try:
                button, _, _ = sequence[3:-1].split(";")
                if button in ("64", "65"):
                    return ("mouse_up", "") if button == "64" else ("mouse_down", "")
            except ValueError:
                pass
        return "ignored", sequence

    def _handle_exit_key(self, key: str) -> bool:
        """Show an exit hint once, then exit when the same control key repeats."""
        if self.exit_notice_key == key and time.monotonic() < self.exit_notice_until:
            return True
        self.exit_notice_key = key
        self.exit_notice = f" Press {key} to exit"
        self.exit_notice_until = time.monotonic() + 3.0
        self.render()
        return False

    def _submit_query(self, query: str):
        """Handle a completed prompt consistently for line and key input."""
        query = query.strip()
        if not query:
            return
        if not self.prompt_history or self.prompt_history[-1] != query:
            self.prompt_history.append(query)
        self.history_index = len(self.prompt_history)

        if query.lower() in ("/exit", "exit", "quit", ":q"):
            self.running = False
            return
        if query.lower() in ("/help", "help", "?"):
            self.write_transcript("\n📖 Commands:")
            self.write_transcript("  /reset  - Clear multi-turn conversation memory and restart fresh")
            self.write_transcript("  /status - Show file count, chunk statistics, and active model")
            self.write_transcript("  /exit   - Quit cn watch and restore terminal\n")
            return
        if query.lower() in ("/reset", "reset", "/clear", "clear"):
            self.on_reset()
            self.tokens_total = 0
            self.tokens_prompt = 0
            self.tokens_completion = 0
            self.turn_count = 0
            self.last_tool_count = 0
            self.write_transcript("\n🧹 Conversation context reset. Started fresh session.\n")
            return
        if query.lower() in ("/status", "status"):
            self.write_transcript(f"\n{self.on_status()}\n")
            return

        theme = detect_terminal_theme()
        divider_color = "\033[38;5;75m" if theme == "dark" else "\033[34m"
        prompt_color = "\033[38;5;229m" if theme == "dark" else "\033[38;5;26m"
        divider = f"{divider_color}{'─' * self.get_dimensions()[0]}\033[0m"
        prompt = f"{prompt_color}👤 You: {query}\033[0m"
        self.write_transcript(divider)
        self.write_transcript(prompt)
        self.write_transcript(divider)
        self.on_submit(query)

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

        input_tty = hasattr(sys.stdin, "fileno") and sys.stdin.isatty()
        input_text = ""
        self.history_index = len(self.prompt_history)
        term_state = None
        try:
            if input_tty:
                import termios
                import tty

                term_state = termios.tcgetattr(sys.stdin.fileno())
                fd = sys.stdin.fileno()
                tty.setcbreak(fd)
                # Keep control keys as input events instead of letting the
                # terminal turn Ctrl-C into SIGINT (or Ctrl-Q into flow
                # control) before the TUI can display its exit hint.
                terminal_attrs = termios.tcgetattr(fd)
                terminal_attrs[0] &= ~termios.IXON
                terminal_attrs[3] &= ~(termios.ECHO | termios.ECHONL | termios.ISIG | termios.IEXTEN)
                termios.tcsetattr(fd, termios.TCSANOW, terminal_attrs)
                sys.stdout.write("\033[?1000h\033[?1006h")
                sys.stdout.flush()

            while self.running:
                self.render(input_text)
                if input_tty:
                    kind, value = self._read_key()
                    if kind == "eof":
                        break
                    if kind == "timeout":
                        continue
                    if kind == "char" and value in ("\003", "\004", "\021"):
                        key = {"\003": "Ctrl-C", "\004": "Ctrl-D", "\021": "Ctrl-Q"}[value]
                        if self._handle_exit_key(key):
                            break
                        continue
                    if kind == "char" and value in ("\r", "\n"):
                        self._submit_query(input_text)
                        input_text = ""
                    elif kind == "char" and value in ("\177", "\b"):
                        input_text = input_text[:-1]
                    elif kind == "char" and value.isprintable():
                        input_text += value
                    elif kind == "up":
                        if self.prompt_history:
                            self.history_index = max(0, self.history_index - 1)
                            input_text = self.prompt_history[self.history_index]
                    elif kind == "down":
                        if self.history_index < len(self.prompt_history) - 1:
                            self.history_index += 1
                            input_text = self.prompt_history[self.history_index]
                        else:
                            self.history_index = len(self.prompt_history)
                            input_text = ""
                    elif kind == "page_up":
                        self.scroll_transcript(max(1, self._viewport_height() - 1))
                    elif kind == "page_down":
                        self.scroll_transcript(-max(1, self._viewport_height() - 1))
                    elif kind == "mouse_up":
                        self.scroll_transcript(3)
                    elif kind == "mouse_down":
                        self.scroll_transcript(-3)
                else:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    self._submit_query(line)
        except (KeyboardInterrupt, EOFError):
            pass
        except Exception as e:
            self.write_transcript(f"\n❌ Error: {e}\n")
        finally:
            if term_state is not None:
                import termios

                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, term_state)
                sys.stdout.write("\033[?1000l\033[?1006l")
            self.exit_screen()
            self.on_exit()
