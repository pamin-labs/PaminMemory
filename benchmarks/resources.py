"""Memory and disk, counted so that arms of different shapes compare.

Summing RSS across a process tree is wrong here, and wrong in a way that
flatters whichever arm forks most: PostgreSQL's backends share one pool of
buffers and RSS charges that pool to every one of them -- 326 MB summed as RSS
against 78 MB summed as PSS, for the same ten processes. PSS divides each
shared page among the processes mapping it, so a tree's PSS is what the tree
really occupies.

Attribution is by process tree, and the tree is not always the obvious one.
`pamin serve` starts PostgreSQL and the cluster outlives it, reparented, so a
walk down from the server returns the server alone and reports a
database-backed system as using no database. The cluster is claimed by its
data directory instead.
"""
import os
import subprocess
import threading


def pss_mb(pids):
    """Proportional set size over a set of pids, in MiB.

    Shared pages are divided among the processes mapping them, so this can be
    summed across a tree without counting PostgreSQL's shared buffers once per
    backend.
    """
    total = 0
    for pid in pids:
        try:
            for line in open(f"/proc/{pid}/smaps_rollup"):
                if line.startswith("Pss:"):
                    total += int(line.split()[1])
                    break
        except (OSError, ValueError):
            continue
    return round(total / 1024)


def children_of(roots):
    """Every pid whose parent chain reaches one of `roots`, and the roots."""
    parent = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            stat = open(f"/proc/{pid}/stat").read()
            parent[int(pid)] = int(stat[stat.rindex(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
    tree = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, ppid in parent.items():
            if ppid in tree and pid not in tree:
                tree.add(pid)
                changed = True
    return tree


def postgres_for(home):
    """The PostgreSQL cluster this workspace started, and its backends.

    Not found by walking children of `pamin serve`: the cluster outlives the
    process that started it and is reparented, so a ppid walk returns the
    server alone and reports a database-backed system as using no database.
    The parent postmaster carries `-D <home>/postgres/data` on its command
    line, and everything under it is a backend of that cluster.
    """
    roots = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if open(f"/proc/{pid}/comm").read().strip() != "postgres":
                continue
            args = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="ignore")
            if f"{home}/postgres/data" in args:
                roots.append(int(pid))
        except OSError:
            continue
    return children_of(roots) if roots else set()


def pids_named(name):
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if open(f"/proc/{pid}/comm").read().strip() == name:
                out.append(int(pid))
        except OSError:
            continue
    return out


class Sampler(threading.Thread):
    """Peak and resting PSS of a tree, sampled while work happens.

    Resting memory is what a server settles at; peak is what the machine has
    to have free. Reporting only the first is how a measurement says a
    workload fits when it does not.
    """

    def __init__(self, resolve, interval=1.0):
        super().__init__(daemon=True)
        self.resolve, self.interval = resolve, interval
        self.samples = []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.wait(self.interval):
            try:
                self.samples.append(pss_mb(self.resolve()))
            except Exception:
                continue

def dir_mb(path):
    """Bytes on disk, in MiB, or zero when the arm keeps none."""
    if not path or not os.path.exists(path):
        return 0
    out = subprocess.run(["du", "-sm", path], capture_output=True, text=True)
    return int(out.stdout.split()[0]) if out.returncode == 0 else 0


def tree_for(arm, pamin_home):
    """The processes an arm's memory should be charged to.

    An arm that runs inside the caller is charged the caller; an arm backed by
    a server is charged the server and its database. The embedder is not
    charged here even when an arm calls out for it, because where that runs is
    an architectural difference and folding it in would hide one.
    """
    if getattr(arm, "holder", None) == "pamin":
        return children_of(pids_named("pamin")) | postgres_for(pamin_home)
    return {os.getpid()}
