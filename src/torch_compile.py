# ---------------------------------------------------------------------------
# Tenacity-powered compile helper
# ---------------------------------------------------------------------------
import os
import torch
from contextlib import contextmanager
import sys
import threading
import io

from tenacity import (
    retry, retry_if_exception, stop_after_attempt, wait_exponential
)

_LOCK_PATTERNS = (
    "unable to acquire lock",
    "file exists",                 # build_dir/lock already there
    "no such file or directory"    # .so missing because other proc quit early
)

def _is_lock_error(exc: Exception) -> bool:
    """Return True for the transient lock-file collisions PyTorch emits."""
    return any(pat in str(exc).lower() for pat in _LOCK_PATTERNS)

@contextmanager
def capture_output():
    """
    Capture *everything* written to stdout/stderr by the current process
    and any child processes during the `with` block.

    Yields
    ------
    buffer : io.StringIO
        Call `buffer.getvalue()` afterwards to obtain the combined log.
    """
    # Flush Python’s own buffers first so we don't lose anything
    sys.stdout.flush()
    sys.stderr.flush()

    # Duplicate the real stdout/stderr so we can restore them later
    orig_out_fd = os.dup(1)
    orig_err_fd = os.dup(2)

    # Create a pipe; r_fd is what we'll read, w_fd replaces 1 & 2
    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 1)          # 1 → pipe
    os.dup2(w_fd, 2)          # 2 → pipe
    os.close(w_fd)            # writer now only referenced by FD 1/2

    buf = io.StringIO()

    # Background thread to drain the read end so the pipe never fills
    def _drain():
        with os.fdopen(r_fd, 'r', encoding='utf-8', errors='replace') as r:
            for line in r:
                buf.write(line)

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    try:
        yield buf            # inside the with-block
    finally:
        # Restore original FDs
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(orig_out_fd, 1)
        os.dup2(orig_err_fd, 2)
        os.close(orig_out_fd)
        os.close(orig_err_fd)
        t.join(timeout=0.1)  # finish draining what’s left

# ---------------------------------------------------------------------------
# 1.  Add this exception so failures always carry their compile log
# ---------------------------------------------------------------------------
class CompilationFailure(RuntimeError):
    def __init__(self, message: str, log_text: str):
        super().__init__(message)
        self.compile_log = log_text


# ---------------------------------------------------------------------------
# 2.  Updated helper: retries on lock errors, returns (model, log) on success,
#     raises CompilationFailure(log=…) on the final non-lock error
# ---------------------------------------------------------------------------
@retry(                                   # Tenacity handles the looping/back-off
    retry=retry_if_exception(_is_lock_error),
    stop=stop_after_attempt(3),           # 1 try + 2 retries
    wait=wait_exponential(multiplier=0.5, max=4),
    reraise=True,                         # propagate non-lock errors to us
)
def _compile_with_log(load_custom_model, src: str, ctx: dict,
                      build_dir: str, device):
    """
    Build a custom CUDA extension once.
    Returns
    -------
    model, log_text      on success
    Raises
    ------
    CompilationFailure   (with .compile_log) on error after retries
    """
    with capture_output() as buf:             # Captures *all* stdout/stderr
        try:
            os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
            os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
            # Use this if you oom
            # os.environ.setdefault("MAX_JOBS", "1")
            model = load_custom_model(src, ctx, build_dir)
            torch.cuda.synchronize(device=device)
            return model, buf.getvalue()      # ✓ success path
        except Exception as exc:
            # We’re still inside the with-block ↴  log is already filled.
            log_txt = buf.getvalue()
            raise CompilationFailure(str(exc), log_txt) from exc

def compile_kernel(load_fn, src, ctx, build_dir, device):
    return _compile_with_log(load_fn, src, ctx, build_dir, device)