"""Fence Windows test subprocesses behind a parent-owned kill-on-close Job.

The bootstrap reads one parent byte before it starts the requested command. A
failed assignment or lost parent closes stdin, so no test/descendant can start
outside the Job. Only the parent owns its uninheritable kernel handle.
"""

from contextlib import contextmanager
import subprocess
import sys


def _job(pid):
    from app.runtime.worker import WorkerError, _WindowsJob

    try:
        return _WindowsJob(pid)
    except WorkerError:
        raise OSError("Could not protect the Windows test process tree") from None


@contextmanager
def owned_process(command, *, cwd, env, stdout):
    process = subprocess.Popen([sys.executable, "-m", "tools.windows_checks", *command],
                               cwd=cwd, env=env, stdin=subprocess.PIPE,
                               stdout=stdout, stderr=subprocess.STDOUT)
    job = None
    try:
        job = _job(process.pid)
        process.stdin.write(b"G")
        process.stdin.close()
        yield process
    finally:
        try:
            if not process.stdin.closed:
                process.stdin.close()
        except OSError:
            pass
        try:
            if job is not None:
                job.close()  # Normal completion also reaps leaked grandchildren.
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


def main(argv=None):
    command = sys.argv[1:] if argv is None else argv
    if not command or sys.stdin.buffer.read(1) != b"G":
        return 72
    return subprocess.call(command, stdin=subprocess.DEVNULL)


if __name__ == "__main__":
    raise SystemExit(main())
