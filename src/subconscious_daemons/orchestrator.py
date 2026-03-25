"""
orchestrator.py — Subconscious Process Orchestrator
The single process you start at logon. Owns all other daemons.

Responsibilities:
  1. PID lockfile singleton — physically impossible to run two orchestrators
  2. Process registry — knows exactly what's running and when it started
  3. Ollama coordination — processes request/release the GPU lock before using LLM
  4. think.py lifecycle — start, monitor, restart on crash
  5. Dream-time scheduler — pause think.py, start overnight.py at 20:00, reverse at 07:00
  6. Status reporting — writes orchestrator_status.json every 30s for health.py

Usage:
  Run once at Windows logon:
    pythonw.exe orchestrator.py
  Or for debugging:
    python.exe orchestrator.py --verbose
"""

import sys
import os
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import time
import ctypes
import atexit
import logging
import argparse
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
from config import SUBCON_DIR, LOGS_DIR, VENV_PYTHON, VENV_PYTHONW

PID_FILE        = LOGS_DIR / "orchestrator.pid"
STATE_FILE      = LOGS_DIR / "process_state.json"
STATUS_FILE     = LOGS_DIR / "orchestrator_status.json"
OLLAMA_LOCK     = LOGS_DIR / "ollama.lock"

DREAM_START_HOUR = 20   # 8 PM — suspend think, start overnight
WAKE_HOUR        = 7    # 7 AM — terminate overnight, resume think

CHECK_INTERVAL   = 30   # seconds between state machine ticks
MAX_THINK_RESTARTS_PER_DAY = 5
MAX_OVERNIGHT_RESTARTS     = 3

# ── Logging ───────────────────────────────────────────────────────────────────
LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOGS_DIR / f"orchestrator-{datetime.now().strftime('%Y-%m-%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [orch] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


# ── Windows process control (no psutil) ──────────────────────────────────────
kernel32 = ctypes.windll.kernel32
THREAD_SUSPEND_RESUME = 0x0002
TH32CS_SNAPTHREAD     = 0x00000004

class THREADENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize",             ctypes.c_ulong),
        ("cntUsage",           ctypes.c_ulong),
        ("th32ThreadID",       ctypes.c_ulong),
        ("th32OwnerProcessID", ctypes.c_ulong),
        ("tpBasePri",          ctypes.c_long),
        ("tpDeltaPri",         ctypes.c_long),
        ("dwFlags",            ctypes.c_ulong),
    ]

def _threads_of(pid: int):
    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    if snap == ctypes.c_void_p(-1).value:
        return []
    threads = []
    te = THREADENTRY32()
    te.dwSize = ctypes.sizeof(THREADENTRY32)
    if kernel32.Thread32First(snap, ctypes.byref(te)):
        while True:
            if te.th32OwnerProcessID == pid:
                threads.append(te.th32ThreadID)
            if not kernel32.Thread32Next(snap, ctypes.byref(te)):
                break
    kernel32.CloseHandle(snap)
    return threads

def suspend_pid(pid: int) -> bool:
    ok = False
    for tid in _threads_of(pid):
        h = kernel32.OpenThread(THREAD_SUSPEND_RESUME, False, tid)
        if h:
            kernel32.SuspendThread(h)
            kernel32.CloseHandle(h)
            ok = True
    return ok

def resume_pid(pid: int) -> bool:
    ok = False
    for tid in _threads_of(pid):
        h = kernel32.OpenThread(THREAD_SUSPEND_RESUME, False, tid)
        if h:
            kernel32.ResumeThread(h)
            kernel32.CloseHandle(h)
            ok = True
    return ok

def pid_alive(pid: int) -> bool:
    """Check if a PID is alive using Windows API."""
    STILL_ACTIVE = 259
    h = kernel32.OpenProcess(0x0400 | 0x0010, False, pid)  # QUERY_INFO + VM_READ
    if not h:
        return False
    code = ctypes.c_ulong(0)
    kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    kernel32.CloseHandle(h)
    return code.value == STILL_ACTIVE

def kill_pid(pid: int):
    """Forcefully terminate a process."""
    h = kernel32.OpenProcess(0x0001, False, pid)  # TERMINATE
    if h:
        kernel32.TerminateProcess(h, 1)
        kernel32.CloseHandle(h)


# ── PID Lockfile ──────────────────────────────────────────────────────────────
def acquire_lock() -> bool:
    """Try to acquire the orchestrator singleton lock.
    Returns True if we are the unique orchestrator, False if another is running."""
    if PID_FILE.exists():
        try:
            existing_pid = int(PID_FILE.read_text().strip())
            if pid_alive(existing_pid) and existing_pid != os.getpid():
                log.warning(f"Orchestrator already running (PID {existing_pid}). Exiting.")
                return False
        except (ValueError, OSError):
            pass  # stale file — overwrite it
    PID_FILE.write_text(str(os.getpid()))
    return True

def release_lock():
    try:
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text().strip())
            if pid == os.getpid():
                PID_FILE.unlink()
    except Exception:
        pass


# ── Ollama Lock ───────────────────────────────────────────────────────────────
_ollama_lock = threading.Lock()  # in-process lock

def acquire_ollama(holder: str, timeout: int = 300) -> bool:
    """Claim the Ollama GPU lock for `holder`. Blocks until available or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with _ollama_lock:
            if not OLLAMA_LOCK.exists():
                OLLAMA_LOCK.write_text(json.dumps({
                    "holder": holder,
                    "acquired": datetime.now().isoformat(),
                    "pid": os.getpid()
                }))
                return True
            # Check if lock holder is stale (holder process died)
            try:
                info = json.loads(OLLAMA_LOCK.read_text())
                acquired_at = datetime.fromisoformat(info["acquired"])
                if datetime.now() - acquired_at > timedelta(minutes=10):
                    log.warning(f"Stale Ollama lock from {info.get('holder')} — stealing it")
                    OLLAMA_LOCK.write_text(json.dumps({
                        "holder": holder,
                        "acquired": datetime.now().isoformat(),
                        "pid": os.getpid()
                    }))
                    return True
            except Exception:
                OLLAMA_LOCK.unlink(missing_ok=True)
                continue
        time.sleep(5)
    return False

def release_ollama(holder: str):
    """Release the Ollama lock if we hold it."""
    try:
        if OLLAMA_LOCK.exists():
            info = json.loads(OLLAMA_LOCK.read_text())
            if info.get("holder") == holder:
                OLLAMA_LOCK.unlink()
    except Exception:
        OLLAMA_LOCK.unlink(missing_ok=True)


# ── Process Registry ──────────────────────────────────────────────────────────
def _load_state() -> dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _save_state(state: dict):
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass

def _write_status(phase: str, detail: str, think_pid: int = 0, overnight_pid: int = 0,
                  think_suspended: bool = False, restarts: int = 0):
    try:
        STATUS_FILE.write_text(json.dumps({
            "orchestrator_pid": os.getpid(),
            "phase": phase,
            "detail": detail,
            "timestamp": datetime.now().isoformat(),
            "think_pid": think_pid,
            "think_suspended": think_suspended,
            "overnight_pid": overnight_pid,
            "think_restarts_today": restarts,
            "ollama_lock": OLLAMA_LOCK.exists(),
        }, indent=2), encoding="utf-8")
    except Exception:
        pass


# ── Process Launchers ─────────────────────────────────────────────────────────
def _update_process_state(think_pid: int = 0, think_status: str = "unknown",
                          overnight_pid: int = 0, overnight_status: str = "off"):
    """Write process state so think.py singleton guard and health.py can read it."""
    try:
        state = {
            "think_pid": think_pid,
            "think_status": think_status,
            "overnight_pid": overnight_pid,
            "overnight_status": overnight_status,
            "orchestrator_pid": os.getpid(),
            "updated": datetime.now().isoformat(),
        }
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass

def launch_think() -> subprocess.Popen:
    log.info("▶️  Launching think.py...")
    proc = subprocess.Popen(
        [VENV_PYTHONW, str(SUBCON_DIR / "think.py")],
        cwd=str(SUBCON_DIR),
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"},
    )
    log.info(f"   think.py started (PID {proc.pid})")
    _update_process_state(think_pid=proc.pid, think_status="running")
    return proc

def launch_overnight(topic: str = None, depth: int = 4) -> subprocess.Popen:
    log.info("🌙 Launching overnight.py...")
    cmd = [VENV_PYTHON, str(SUBCON_DIR / "overnight.py"), "--depth", str(depth)]
    if topic:
        cmd += ["--topic", topic]
    proc = subprocess.Popen(
        cmd,
        cwd=str(SUBCON_DIR),
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log.info(f"   overnight.py started (PID {proc.pid})")
    _update_process_state(overnight_pid=proc.pid, overnight_status="running")
    return proc

def terminate_proc(proc: subprocess.Popen, name: str, timeout: int = 15):
    if proc is None or proc.poll() is not None:
        return
    log.info(f"⏹️  Terminating {name} (PID {proc.pid})...")
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log.warning(f"   {name} didn't terminate gracefully, killing...")
        proc.kill()
        proc.wait(timeout=5)


# ── Time Helpers ──────────────────────────────────────────────────────────────
def is_dream_time() -> bool:
    h = datetime.now().hour
    return h >= DREAM_START_HOUR or h < WAKE_HOUR


# ── State Machine ─────────────────────────────────────────────────────────────
class Orchestrator:
    def __init__(self):
        self.think_proc: subprocess.Popen | None = None
        self.overnight_proc: subprocess.Popen | None = None
        self.think_suspended = False
        self.dream_phase = False
        self.think_restarts_today = 0
        self.overnight_restarts = 0
        self._restart_date = datetime.now().date()

    def _reset_daily_counters(self):
        today = datetime.now().date()
        if today != self._restart_date:
            self.think_restarts_today = 0
            self.overnight_restarts = 0
            self._restart_date = today

    def _think_alive(self) -> bool:
        return (self.think_proc is not None
                and self.think_proc.poll() is None
                and pid_alive(self.think_proc.pid))

    def _overnight_alive(self) -> bool:
        return (self.overnight_proc is not None
                and self.overnight_proc.poll() is None
                and pid_alive(self.overnight_proc.pid))

    # ── Day phase: think.py running ──────────────────────────────────────────
    def enter_day_phase(self):
        if self.dream_phase:
            log.info("☀️  Entering DAY phase — terminating overnight, resuming think")
            self.dream_phase = False

        # Kill overnight if somehow still running
        if self._overnight_alive():
            terminate_proc(self.overnight_proc, "overnight.py")
            self.overnight_proc = None

        # Resume think if suspended
        if self.think_suspended and self._think_alive():
            log.info(f"▶️  Resuming think.py (PID {self.think_proc.pid})")
            resume_pid(self.think_proc.pid)
            self.think_suspended = False

        # Start think if not running
        if not self._think_alive():
            self._start_think()

    def _start_think(self):
        self._reset_daily_counters()
        if self.think_restarts_today >= MAX_THINK_RESTARTS_PER_DAY:
            log.error(f"❌ think.py has crashed {self.think_restarts_today}x today. Not restarting.")
            return
        self.think_proc = launch_think()
        self.think_suspended = False
        if self.think_restarts_today > 0:
            log.info(f"   Restart #{self.think_restarts_today} today")
        self.think_restarts_today += 1
        time.sleep(3)  # brief settle time

    # ── Dream phase: overnight.py running, think.py suspended ────────────────
    def enter_dream_phase(self):
        if not self.dream_phase:
            log.info("🌙 Entering DREAM phase — suspending think, starting overnight")
            self.dream_phase = True
            self.overnight_restarts = 0

        # Suspend think.py to free Ollama GPU
        if self._think_alive() and not self.think_suspended:
            log.info(f"⏸️  Suspending think.py (PID {self.think_proc.pid})")
            suspend_pid(self.think_proc.pid)
            self.think_suspended = True
            time.sleep(2)  # let any in-flight Ollama call drain

        # Start overnight if not running
        if not self._overnight_alive():
            if self.overnight_restarts >= MAX_OVERNIGHT_RESTARTS:
                log.error(f"❌ overnight.py restarted {self.overnight_restarts}x. Giving up.")
                return
            self.overnight_proc = launch_overnight()
            self.overnight_restarts += 1

    # ── Main tick ─────────────────────────────────────────────────────────────
    def tick(self):
        self._reset_daily_counters()

        if is_dream_time():
            self.enter_dream_phase()

            # Check for stalled overnight
            if self._overnight_alive():
                pass  # overnight.py manages it own stall detection
            elif self.overnight_proc is not None and self.overnight_proc.poll() is not None:
                rc = self.overnight_proc.returncode
                if rc != 0:
                    log.warning(f"overnight.py exited with code {rc}. Restarting...")
                else:
                    log.info("overnight.py completed normally.")
                    self.overnight_proc = None
                    # If still dream time, restart it
                    if is_dream_time():
                        self.enter_dream_phase()
        else:
            self.enter_day_phase()

            # Check think.py health
            if not self._think_alive() and not self.think_suspended:
                if self.think_proc is not None:
                    rc = self.think_proc.poll()
                    log.warning(f"think.py died (exit {rc}). Restarting...")
                self._start_think()

        # Write status files
        think_pid = self.think_proc.pid if self.think_proc else 0
        overnight_pid = self.overnight_proc.pid if self.overnight_proc else 0
        think_status = "suspended" if self.think_suspended else ("running" if self._think_alive() else "dead")
        overnight_status = "running" if self._overnight_alive() else "off"

        _write_status(
            phase="dreaming" if self.dream_phase else "thinking",
            detail=f"think={think_status}, overnight={overnight_status}",
            think_pid=think_pid,
            overnight_pid=overnight_pid,
            think_suspended=self.think_suspended,
            restarts=self.think_restarts_today - 1,
        )
        _update_process_state(
            think_pid=think_pid, think_status=think_status,
            overnight_pid=overnight_pid, overnight_status=overnight_status,
        )

    def shutdown(self):
        log.info("🛑 Orchestrator shutting down...")
        # Resume think.py before exiting so it can clean up
        if self.think_suspended and self._think_alive():
            resume_pid(self.think_proc.pid)
        terminate_proc(self.overnight_proc, "overnight.py")
        # Don't kill think.py on shutdown — let it keep running
        # (only the orchestrator exits, think.py continues)
        release_ollama("orchestrator")
        release_lock()
        log.info("   Shutdown complete.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Subconscious Orchestrator")
    parser.add_argument("--verbose", action="store_true", help="Extra logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    log.info("=" * 60)
    log.info("  🧠 SUBCONSCIOUS ORCHESTRATOR STARTING")
    log.info(f"  PID: {os.getpid()}")
    log.info(f"  Dream hours: {DREAM_START_HOUR}:00 – {WAKE_HOUR}:00")
    log.info("=" * 60)

    # Singleton check
    if not acquire_lock():
        sys.exit(1)

    orch = Orchestrator()
    atexit.register(orch.shutdown)

    try:
        while True:
            try:
                orch.tick()
            except Exception as e:
                log.error(f"Tick error (continuing): {e}", exc_info=True)
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        log.info("Interrupted by user.")

    orch.shutdown()


if __name__ == "__main__":
    main()
