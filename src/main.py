#!/usr/bin/env python3
"""Myers' diff.

    python main.py lines     FILE_A FILE_B    # Part A: minimal line diff
    python main.py highlight FILE_A FILE_B    # Part B: Part A + changed characters

WHERE TO CHANGE THINGS
    how a file is split into lines ........ read_lines()
    the diff algorithm .................... middle_snake() and myers()
    speed-ups (trim / filter / int ids) ... diff_flags()
    how ranges like 3-5,9-12 are written .. ranges()
    which '-' line pairs with which '+' ... print_diff()
    exit codes and arguments .............. main()
"""
import sys


# ---------------------------------------------------------------- 1. read
def read_lines(path):
    """Raw bytes, split on b"\\n", keep any b"\\r", a final newline adds no line."""
    with open(path, "rb") as f:
        lines = f.read().split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    return lines


# ------------------------------------------------------- 2. Myers' algorithm
# Edit graph: x = position in A, y = position in B, diagonal k = x - y.
#   diagonal step (x+1, y+1): lines equal, free      -> keep
#   right step    (x+1, y)  : delete a line of A     -> "-"   cost 1
#   down step     (x, y+1)  : insert a line of B     -> "+"   cost 1
# fwd[k] = furthest x reached on diagonal k, searching from the top-left.
# bwd[k] = smallest x reached on diagonal k, searching from the bottom-right.
# Both searches grow one edit at a time and stop when they touch; the touching
# point lies on a shortest path, so we split there and solve the two halves.
# (Linear space: only the two arrays fwd and bwd exist.)

def middle_snake(A, B, a0, a1, b0, b1, fwd, bwd, big):
    """A point (x, y) on a shortest path through A[a0:a1] x B[b0:b1]."""
    dmin, dmax = a0 - b1, a1 - b0        # lowest / highest diagonal in the box
    fk, bk = a0 - b0, a1 - b1            # start diagonals of the two searches
    odd = (fk - bk) & 1                  # which search can see the meeting
    fmin = fmax = fk
    bmin = bmax = bk
    fwd[fk] = a0
    bwd[bk] = a1
    while True:
        # ---- forward search: one more edit, so one more diagonal on each side
        if fmin > dmin:
            fmin -= 1
            fwd[fmin - 1] = -1
        else:
            fmin += 1
        if fmax < dmax:
            fmax += 1
            fwd[fmax + 1] = -1
        else:
            fmax -= 1
        for k in range(fmax, fmin - 1, -2):
            # come from the neighbour that got furthest (right or down step)
            x = fwd[k - 1] + 1 if fwd[k - 1] >= fwd[k + 1] else fwd[k + 1]
            y = x - k
            while x < a1 and y < b1 and A[x] == B[y]:     # the snake
                x += 1
                y += 1
            fwd[k] = x
            if odd and bmin <= k <= bmax and bwd[k] <= x:  # searches met
                return x, y
        # ---- backward search: mirror image
        if bmin > dmin:
            bmin -= 1
            bwd[bmin - 1] = big
        else:
            bmin += 1
        if bmax < dmax:
            bmax += 1
            bwd[bmax + 1] = big
        else:
            bmax -= 1
        for k in range(bmax, bmin - 1, -2):
            x = bwd[k - 1] if bwd[k - 1] < bwd[k + 1] else bwd[k + 1] - 1
            y = x - k
            while x > a0 and y > b0 and A[x - 1] == B[y - 1]:   # snake, backwards
                x -= 1
                y -= 1
            bwd[k] = x
            if not odd and fmin <= k <= fmax and x <= fwd[k]:   # searches met
                return x, y


def myers(A, B):
    """Minimal diff of two lists. Returns (flagsA, flagsB): 1 = changed, 0 = kept."""
    n, m = len(A), len(B)
    flagsA, flagsB = bytearray(n), bytearray(m)
    fwd = [0] * (n + m + 3)
    bwd = [0] * (n + m + 3)
    todo = [(0, n, 0, m)]                       # boxes still to solve (no recursion)
    while todo:
        a0, a1, b0, b1 = todo.pop()
        while a0 < a1 and b0 < b1 and A[a0] == B[b0]:          # equal start: keep
            a0 += 1
            b0 += 1
        while a0 < a1 and b0 < b1 and A[a1 - 1] == B[b1 - 1]:  # equal end: keep
            a1 -= 1
            b1 -= 1
        if a0 == a1:                                  # only inserts are left
            flagsB[b0:b1] = b"\x01" * (b1 - b0)
        elif b0 == b1:                                # only deletes are left
            flagsA[a0:a1] = b"\x01" * (a1 - a0)
        else:
            x, y = middle_snake(A, B, a0, a1, b0, b1, fwd, bwd, n + m + 5)
            todo.append((a0, x, b0, y))
            todo.append((x, a1, y, b1))
    return flagsA, flagsB


# ------------------------------------------------------------ 3. speed-ups
def diff_flags(A, B):
    """Same result as myers(A, B), but much faster on big inputs.

    Used for lines (Part A) and for characters of one line (Part B).
    """
    n, m = len(A), len(B)
    flagsA, flagsB = bytearray(n), bytearray(m)
    # 1. trim equal lines at the start and at the end (they are always kept)
    lo = 0
    while lo < n and lo < m and A[lo] == B[lo]:
        lo += 1
    endA, endB = n, m
    while endA > lo and endB > lo and A[endA - 1] == B[endB - 1]:
        endA -= 1
        endB -= 1
    if lo == endA or lo == endB:                      # one side is used up
        flagsA[lo:endA] = b"\x01" * (endA - lo)
        flagsB[lo:endB] = b"\x01" * (endB - lo)
        return flagsA, flagsB
    midA, midB = A[lo:endA], B[lo:endB]
    # 2. a line that is not in the other file can never be kept: it is changed
    inA, inB = set(midA), set(midB)
    posA = [i for i, x in enumerate(midA) if x in inB]
    posB = [j for j, x in enumerate(midB) if x in inA]
    flagsA[lo:endA] = b"\x01" * (endA - lo)           # changed until proven kept
    flagsB[lo:endB] = b"\x01" * (endB - lo)
    # 3. small integers compare faster than byte strings
    ids = {}
    seqA = [ids.setdefault(midA[i], len(ids)) for i in posA]
    seqB = [ids[midB[j]] for j in posB]
    # 4. run Myers on what is left, then mark the kept lines
    keepA, keepB = myers(seqA, seqB)
    for r, i in enumerate(posA):
        if not keepA[r]:
            flagsA[lo + i] = 0
    for r, j in enumerate(posB):
        if not keepB[r]:
            flagsB[lo + j] = 0
    return flagsA, flagsB


# ------------------------------------------------------------- 4. Part B
def ranges(flags):
    """Flags 0/1 -> "3-5,9-12" (end not included), or "." when nothing changed."""
    parts = []
    i = flags.find(1)
    while i != -1:
        j = flags.find(0, i)
        if j == -1:
            j = len(flags)
        parts.append("%d-%d" % (i, j))
        i = flags.find(1, j)
    return ",".join(parts) or "."


def highlight_line(old, new):
    """The '? old-ranges | new-ranges' line for one pair of lines (by characters)."""
    old = old.decode("utf-8", "surrogateescape")     # 1 code point = 1 character
    new = new.decode("utf-8", "surrogateescape")
    fo, fn = diff_flags(old, new)
    return ("? %s | %s\n" % (ranges(fo), ranges(fn))).encode("ascii")


# ------------------------------------------------------------- 5. output
def print_diff(A, B, flagsA, flagsB, highlight):
    n, m = len(A), len(B)
    out = []
    i = j = 0
    while i < n or j < m:
        # end of the run of changed lines that starts here (find 0 = first kept line)
        di = flagsA.find(0, i)
        di = n if di < 0 else di
        dj = flagsB.find(0, j)
        dj = m if dj < 0 else dj
        if di > i or dj > j:
            # a change block: all '-' lines first, then all '+' lines
            dels, adds = A[i:di], B[j:dj]
            if dels:
                out.append(b"-" + b"\n-".join(dels) + b"\n")
            if highlight:
                for p, line in enumerate(adds):
                    out.append(b"+" + line + b"\n")
                    if p < len(dels):                # p-th '-' pairs with p-th '+'
                        out.append(highlight_line(dels[p], line))
            elif adds:
                out.append(b"+" + b"\n+".join(adds) + b"\n")
            i, j = di, dj
        else:
            # a run of kept lines (find 1 = next changed line)
            ka = flagsA.find(1, i)
            kb = flagsB.find(1, j)
            run = min(n - i if ka < 0 else ka - i, m - j if kb < 0 else kb - j)
            out.append(b" " + b"\n ".join(A[i:i + run]) + b"\n")
            i += run
            j += run
    sys.stdout.buffer.write(b"".join(out))


# --------------------------------------------------------------- 6. main
def main(argv):
    if len(argv) != 4 or argv[1] not in ("lines", "highlight"):
        sys.stderr.write("usage: main.py lines|highlight FILE_A FILE_B\n")
        return 2
    try:
        A, B = read_lines(argv[2]), read_lines(argv[3])
    except OSError as e:
        sys.stderr.write("error: cannot read file: %s\n" % e)
        return 2                                     # nothing printed on stdout
    flagsA, flagsB = diff_flags(A, B)
    print_diff(A, B, flagsA, flagsB, argv[1] == "highlight")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))