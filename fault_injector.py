"""
fault_injector.py

Deliberately breaks a real Ollama endpoint on command, so degradation can be
demonstrated reliably instead of hoping a real endpoint happens to fail
during a 3-minute recording.

Design choice: this finds the REAL OS process listening on a given port and
kills or suspends it. This is not a simulated delay inserted into our own
code — the agent genuinely loses the connection or genuinely experiences a
stalled process, the same as it would with any real infrastructure failure.
This is called out explicitly in the README so it's not misrepresented as
organic failure.

Usage (from a separate terminal, WHILE agent.py / demo.py is running a query):
    python fault_injector.py kill 11437       # hard-kills the process (simulates connection drop)
    python fault_injector.py suspend 11437    # pauses the process (simulates a severe stall, still "alive")
    python fault_injector.py resume 11437     # un-pauses a suspended process
"""

import sys
import psutil # type: ignore


def find_pid_by_port(port: int):
    """Finds the PID of whatever process is listening on the given port."""
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
            return conn.pid
    return None


def kill_endpoint(port: int):
    """Hard-kills the process on this port. Simulates a real connection drop —
    the agent's stream_generate() will raise ConnectionDropped."""
    pid = find_pid_by_port(port)
    if pid is None:
        print(f"No process found listening on port {port}.")
        return
    proc = psutil.Process(pid)
    proc.kill()
    print(f"Killed process {pid} on port {port}. Endpoint is now fully down.")


def suspend_endpoint(port: int):
    """Pauses the process without killing it. The connection stays technically
    open, but nothing gets processed — simulates a severe stall (slow but
    alive), not a hard failure."""
    pid = find_pid_by_port(port)
    if pid is None:
        print(f"No process found listening on port {port}.")
        return
    proc = psutil.Process(pid)
    proc.suspend()
    print(f"Suspended process {pid} on port {port}. Endpoint will now hang, not fail outright.")


def resume_endpoint(port: int):
    """Un-pauses a previously suspended process."""
    pid = find_pid_by_port(port)
    if pid is None:
        print(f"No process found listening on port {port}.")
        return
    proc = psutil.Process(pid)
    proc.resume()
    print(f"Resumed process {pid} on port {port}.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fault_injector.py [kill|suspend|resume] <port>")
        sys.exit(1)

    action, port_str = sys.argv[1], sys.argv[2]
    port = int(port_str)

    if action == "kill":
        kill_endpoint(port)
    elif action == "suspend":
        suspend_endpoint(port)
    elif action == "resume":
        resume_endpoint(port)
    else:
        print(f"Unknown action '{action}'. Use kill, suspend, or resume.")
        sys.exit(1)
