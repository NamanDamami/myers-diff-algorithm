#!/usr/bin/env python3
"""Myers' diff (Eugene W. Myers, "An O(ND) Difference Algorithm", 1986).

    python main.py lines     FILE_A FILE_B     # Part A: minimal line diff
    python main.py highlight FILE_A FILE_B     # Part B: Part A + char ranges

Part A prints the whole file pair as an edit script: one prefix character
(" ", "-", "+") in front of every line of A or B.  Part B prints exactly the
same lines and, after every paired changed line, one "? old | new" line that
names the character ranges which differ inside the two lines.

Exit status is 0 on success and 2 when the arguments are wrong or when a file
cannot be read.  Only the diff ever goes to stdout; complaints go to stderr.

The algorithm below is written from scratch and uses nothing but the
standard library's most basic modules (sys).  There is no standard library
diff helper and no external process anywhere in this file.
"""
import sys

# ==========================================================================
# 1. Reading the input files
# ==========================================================================


def read_lines(path):
    """Return the lines of `path` as a list of byte strings.

    The file is opened in binary mode, so the operating system hands us the
    bytes unchanged: text mode would silently rewrite CRLF into LF and make
    b"a\\r" compare equal to b"a", which would be wrong here.

    The rules for turning a byte stream into lines are:
      * split on the single byte b"\\n";
      * drop a trailing empty piece, because a last newline does not start a
        new line (this also makes an empty file produce an empty list);
      * keep every other byte, including CR, as part of the line.

    Examples:
        b""              -> []
        b"\\n"            -> [b""]
        b"a"  / b"a\\n"   -> [b"a"]
        b"a\\n\\nb"        -> [b"a", b"", b"b"]
        b"a\\r\\nb\\r\\n"  -> [b"a\\r", b"b\\r"]
    """
    with open(path, "rb") as handle:
        data = handle.read()
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    return lines


# ==========================================================================
# 2. The edit graph and Myers' algorithm
# ==========================================================================
#
# Think of a rectangle whose x axis is file A and whose y axis is file B.
# A path from the top left corner (0,0) to the bottom right corner (N,M)
# walks over the lines of both files and may use three kinds of steps:
#
#     (x+1, y+1)   diagonal step   A[x] == B[y]  -> the line is kept, cost 0
#     (x+1, y)     right step      only in A      -> "-" (delete),  cost 1
#     (x, y+1)     down step       only in B      -> "+" (insert), cost 1
#
# Any path is an edit script, and a path of cost D means D deletions plus D
# insertions.  Along a path the diagonal steps form a common subsequence of
# the two files, so the cheapest path is the one that keeps the longest
# common subsequence: minimality of a diff is exactly maximality of its LCS.
#
# Diagonals.  Number the diagonal of a point by k = x - y.  A diagonal step
# stays on the same diagonal, an edit step moves to a neighbouring one, so
# spending d edits can only reach diagonals fmid-d .. fmid+d.
#
# V arrays.  Forward search: after d edits, V[d][k] is the furthest x that
# can be reached on diagonal k.  The recurrence is
#
#     V[d][k] = max(V[d-1][k-1] + 1,    # arrive with a right step
#                   V[d-1][k+1])        # arrive with a down step
#
# and each value is then pushed along the diagonal for as long as the two
# files hold equal lines; that run of diagonal steps is the "snake".
#
#     k        k-1  k  k+1        (each bracket holds one k of one d)
#       \      |    |    /
#        \     |    |   /
#         \    |    |  /
#          \   |    | /
#           (0,0)
#
# Backward search: the mirror image, started from (N,M) and walking towards
# (0,0).  W[d][k] is the smallest x still reachable on diagonal k, and
#
#     W[d][k] = min(W[d-1][k-1],      # undo a delete (x grows by one)
#                   W[d-1][k+1] - 1)  # undo an insert
#
# followed by a snake that walks backwards while the lines are equal.
#
# Why this is linear space: storing one V row per d would need O(N*D)
# memory.  Instead we run the forward search and the backward search side by
# side, each with a single array, and stop as soon as they touch.  A point
# where they touch lies on an optimal path, so the rectangle is split there
# and the two halves are solved the same way.  Only two arrays of size N+M
# are alive at any moment, and an explicit stack replaces the recursion.
#
# Why the two searches may stop at the first touch: after the forward search
# has spent d edits it can reach diagonal k for every k with the right
# parity; the backward search has spent d-1 edits (when the check happens in
# the forward pass) or d edits (when it happens in the backward pass).  A
# point reachable by both halves joins the two partial paths into a path of
# cost d + (d-1) = 2d-1 or d + d = 2d.  The optimal cost has the parity of
# N+M, so only one of the two kinds of meeting can ever be optimal, and that
# is what the flag `odd` below selects.

_KEEP = 0        # a kept line: present in both files
_CHANGE = 1     # a deleted line of A / an inserted line of B


def middle_snake(A, B, off1, lim1, off2, lim2, v_fwd, v_bwd, big):
    """Return a point (i1, i2) on a shortest path through A x B.

    The sub-problem is the rectangle A[off1:lim1] x B[off2:lim2].  The two
    searches run together and return as soon as they meet; the caller then
    recurses (iteratively) into A[off1:i1] x B[off2:i2] and
    A[i1:lim1] x B[i2:lim2].

    v_fwd / v_bwd are the two V arrays, indexed by the absolute diagonal
    k = x - y.  Negative diagonals are addressed with Python's negative
    indices; both arrays are long enough (N+M+3) that a negative and a
    positive diagonal can never share a slot.  Unreachable diagonals hold
    -1 in the forward array and `big` (an impossible x) in the backward one.
    """
    # Extremes of the diagonal range inside this rectangle, and the diagonal
    # each search starts on.
    dmin = off1 - lim2            # lowest diagonal of the rectangle
    dmax = lim1 - off2            # highest diagonal of the rectangle
    fmid = off1 - off2            # forward search starts on fmid
    bmid = lim1 - lim2            # backward search starts on bmid
    # A path has the same parity as N+M, so only one pass can meet the other
    # half way; `odd` selects the pass that reports the meeting.
    odd = (fmid - bmid) & 1
    fmin = fmax = fmid            # diagonals known to the forward search
    bmin = bmax = bmid            # diagonals known to the backward search
    v_fwd[fmid] = off1
    v_bwd[bmid] = lim1
    while True:
        # ---- one forward pass: after one more edit each side can be one
        # ---- diagonal further out, so the known range grows by one
        if fmin > dmin:
            fmin -= 1
            v_fwd[fmin - 1] = -1       # the diagonal below fmin is unreachable
        else:
            fmin += 1
        if fmax < dmax:
            fmax += 1
            v_fwd[fmax + 1] = -1       # the diagonal above fmax is unreachable
        else:
            fmax -= 1
        d = fmax
        while d >= fmin:
            # Come from whichever neighbour reached this diagonal furthest:
            # k-1 arrives with a right step, k+1 arrives with a down step.
            left = v_fwd[d - 1]
            right = v_fwd[d + 1]
            if left >= right:
                i1 = left + 1
            else:
                i1 = right
            i2 = i1 - d                  # y follows from the diagonal number
            # Follow the snake: equal lines cost nothing, so keep going.
            while i1 < lim1 and i2 < lim2 and A[i1] == B[i2]:
                i1 += 1
                i2 += 1
            v_fwd[d] = i1
            # Overlap with the backward search (which is one step behind):
            # v_bwd[d] <= i1 means both searches can meet on diagonal d.
            if odd and bmin <= d <= bmax and v_bwd[d] <= i1:
                return i1, i2
            d -= 2                       # only diagonals of this parity
        # ---- one backward pass, the mirror image of the forward one
        if bmin > dmin:
            bmin -= 1
            v_bwd[bmin - 1] = big
        else:
            bmin += 1
        if bmax < dmax:
            bmax += 1
            v_bwd[bmax + 1] = big
        else:
            bmax -= 1
        d = bmax
        while d >= bmin:
            left = v_bwd[d - 1]
            right = v_bwd[d + 1]
            if left < right:
                i1 = left
            else:
                i1 = right - 1
            i2 = i1 - d
            # Snake, now walking backwards along the diagonal.
            while i1 > off1 and i2 > off2 and A[i1 - 1] == B[i2 - 1]:
                i1 -= 1
                i2 -= 1
            v_bwd[d] = i1
            # The forward search is one step ahead here, so this meeting
            # costs d + d, i.e. an even total.
            if not odd and fmin <= d <= fmax and i1 <= v_fwd[d]:
                return i1, i2
            d -= 2


def myers_flags(A, B):
    """Minimal diff of two lists of hashable items (here: small integers).

    Returns (cA, cB): byte arrays of len(A) and len(B) holding _CHANGE for
    every deleted / inserted item and _KEEP for every kept one.
    """
    n = len(A)
    m = len(B)
    cA = bytearray(n)
    cB = bytearray(m)
    # The two V arrays; index k is the diagonal, so the size covers every
    # diagonal from -(m) to n.  big is an x value no real path can reach.
    v_fwd = [0] * (n + m + 3)
    v_bwd = [0] * (n + m + 3)
    big = n + m + 5
    # Sub-problems still to solve, as (off1, lim1, off2, lim2).  A stack is
    # used instead of recursion so that a deep nesting cannot blow up.
    stack = [(0, n, 0, m)]
    while stack:
        off1, lim1, off2, lim2 = stack.pop()
        # Equal items at the start and at the end of the rectangle are always
        # part of some longest common subsequence, so they are kept for free.
        # This costs nothing and often removes most of the rectangle.
        while off1 < lim1 and off2 < lim2 and A[off1] == B[off2]:
            off1 += 1
            off2 += 1
        while off1 < lim1 and off2 < lim2 and A[lim1 - 1] == B[lim2 - 1]:
            lim1 -= 1
            lim2 -= 1
        if off1 == lim1:
            # A is used up: whatever is left of B can only be inserted.
            if off2 < lim2:
                cB[off2:lim2] = b"\x01" * (lim2 - off2)
            continue
        if off2 == lim2:
            # B is used up: whatever is left of A can only be deleted.
            cA[off1:lim1] = b"\x01" * (lim1 - off1)
            continue
        i1, i2 = middle_snake(A, B, off1, lim1, off2, lim2,
                              v_fwd, v_bwd, big)
        # The meeting point splits the rectangle into two smaller ones.  The
        # second half is pushed first so that the first half is solved first,
        # which keeps the output order natural.
        stack.append((off1, i1, off2, i2))
        stack.append((i1, lim1, i2, lim2))
    return cA, cB


# ==========================================================================
# 3. Turning the answer into flags: trimming and filtering
# ==========================================================================


def diff_flags(A, B):
    """Minimal diff of two sequences; 1 = changed, 0 = kept.

    The same routine serves Part A (items are byte strings, whole lines) and
    Part B (items are one-character strings, characters of a line).

    Returns (cA, cB), two byte arrays of the same length as A and B.
    """
    n = len(A)
    m = len(B)
    cA = bytearray(n)
    cB = bytearray(m)

    # ---- step 1: strip the common prefix and the common suffix.
    # The first equal items and the last equal items belong to some longest
    # common subsequence (a common subsequence can always be shifted outwards),
    # so forcing them to be kept cannot make the diff worse; it usually leaves
    # a much smaller rectangle for Myers.
    lo = 0
    limit = n if n < m else m
    while lo < limit and A[lo] == B[lo]:
        lo += 1
    hiA, hiB = n, m
    while hiA > lo and hiB > lo and A[hiA - 1] == B[hiB - 1]:
        hiA -= 1
        hiB -= 1
    if lo == hiA:
        # A is fully consumed by the common prefix and suffix.
        if lo < hiB:
            cB[lo:hiB] = b"\x01" * (hiB - lo)
        return cA, cB
    if lo == hiB:
        cA[lo:hiA] = b"\x01" * (hiA - lo)
        return cA, cB

    midA = A[lo:hiA]
    midB = B[lo:hiB]

    # ---- step 2: drop the lines that cannot possibly be kept.
    # A line of A that does not occur anywhere in B can never be matched, so
    # it is deleted for sure; symmetrically for B.  Removing such lines from
    # both sides keeps the diff minimal: every common subsequence only uses
    # lines that occur on both sides, so the longest common subsequence of the
    # two full sequences is exactly the longest common subsequence of the two
    # trimmed ones.  The work then drops to the lines that really take part in
    # a match, which is what keeps big files fast.
    in_b = set(midB)
    in_a = set(midA)
    # Everything inside the rectangle is changed until Myers says otherwise.
    cA[lo:hiA] = b"\x01" * (hiA - lo)
    cB[lo:hiB] = b"\x01" * (hiB - lo)

    # ---- step 3: replace the surviving lines by small integers.  Comparing
    # two ints is much cheaper than comparing two byte strings of any length,
    # and the integer lists are what the search loop touches.
    ids = {}
    get_id = ids.get
    posA = []
    seqA = []
    add_posA = posA.append
    add_seqA = seqA.append
    for i in range(len(midA)):
        item = midA[i]
        if item in in_b:
            value = get_id(item)
            if value is None:
                value = len(ids)
                ids[item] = value
            add_posA(lo + i)
            add_seqA(value)
    posB = []
    seqB = []
    add_posB = posB.append
    add_seqB = seqB.append
    for j in range(len(midB)):
        item = midB[j]
        if item in in_a:
            value = get_id(item)
            if value is None:
                value = len(ids)
                ids[item] = value
            add_posB(lo + j)
            add_seqB(value)
    del in_a, in_b, ids, midA, midB, get_id       # release before Myers

    # ---- step 4: Myers on the integers, then map the answer back to the
    # ---- original positions.
    fA, fB = myers_flags(seqA, seqB)
    for r in range(len(posA)):
        if not fA[r]:
            cA[posA[r]] = _KEEP
    for r in range(len(posB)):
        if not fB[r]:
            cB[posB[r]] = _KEEP
    return cA, cB


# ==========================================================================
# 4. Part B: the character ranges inside a changed pair of lines
# ==========================================================================


def format_ranges(flags):
    """Turn a 0/1 byte array into b"3-7,9-12" (or b"." when nothing changed).

    A range is written start-end with the end *not* included, ranges keep the
    order of the line, never overlap and touch-ranges are merged because they
    are produced by a single scan of a single run of set flags.
    """
    parts = []
    start = -1
    for index in range(len(flags)):
        if flags[index]:
            if start < 0:
                start = index
        elif start >= 0:
            parts.append(b"%d-%d" % (start, index))
            start = -1
    if start >= 0:
        parts.append(b"%d-%d" % (start, len(flags)))
    if not parts:
        return b"."
    return b",".join(parts)


def highlight_line(old, new):
    """Return the b"? old-ranges | new-ranges\\n" line for one pair of lines.

    The very same minimal diff is run on the *characters* of the two lines.
    Deleting from both lines the characters named by the answer leaves the
    same text on both sides, and the number of named characters is
    len(old) + len(new) - 2 * LCS(old, new), which is the smallest possible.

    The lines are decoded as UTF-8 so that one emoji is one character and
    positions count code points.  A CR at the end of a line is an ordinary
    character and is counted like one.  The "surrogateescape" error handler
    is only a safety net: the exercise guarantees valid UTF-8 here, and even
    if it did not, the ranges would stay consistent.
    """
    text_old = old.decode("utf-8", "surrogateescape")
    text_new = new.decode("utf-8", "surrogateescape")
    cA, cB = diff_flags(list(text_old), list(text_new))
    return b"? " + format_ranges(cA) + b" | " + format_ranges(cB) + b"\n"


# ==========================================================================
# 5. Rendering the output
# ==========================================================================


def build_output(A, B, cA, cB, highlight):
    """Render the edit script as a list of byte strings.

    The walk below is the whole formatting rule book:
      * a run of deleted lines is printed first, a run of inserted lines
        second, so every change block has all of its "-" before all of its
        "+";
      * a kept line is printed with a leading space;
      * in Part B, the p-th "-" line of a block is paired with the p-th "+"
        line of the same block and the range line follows that "+" line.

    Runs of lines are joined with b"\\n " and friends, so the number of byte
    string objects stays proportional to the number of blocks instead of the
    number of lines; nothing is ever copied inside a loop.
    """
    n = len(A)
    m = len(B)
    out = []
    add = out.append
    i = 0                    # next line of A
    j = 0                    # next line of B
    while i < n or j < m:
        # ---- deletions of this block
        deleted = ()
        if i < n and cA[i]:
            start = i
            i += 1
            while i < n and cA[i]:
                i += 1
            deleted = A[start:i]
        # ---- insertions of this block
        inserted = ()
        if j < m and cB[j]:
            start = j
            j += 1
            while j < m and cB[j]:
                j += 1
            inserted = B[start:j]
        if deleted or inserted:
            if deleted:
                add(b"-" + b"\n-".join(deleted) + b"\n")
            if inserted:
                if highlight:
                    # The range line belongs directly under the "+" line it
                    # describes.  Pair by position inside the block: the p-th
                    # "-" line goes with the p-th "+" line, and the lines of
                    # the longer side that have no partner get no range line.
                    for p in range(len(inserted)):
                        add(b"+" + inserted[p] + b"\n")
                        if p < len(deleted):
                            add(highlight_line(deleted[p], inserted[p]))
                else:
                    add(b"+" + b"\n+".join(inserted) + b"\n")
            continue
        # ---- a run of kept lines
        start = i
        while i < n and j < m and not cA[i] and not cB[j]:
            i += 1
            j += 1
        if i == start:        # safety net: always make progress
            i += 1
            j += 1
        add(b" " + b"\n ".join(A[start:i]) + b"\n")
    return out


# ==========================================================================
# 6. Command line
# ==========================================================================


def usage_error(message):
    """Complain on stderr and return the exit status; stdout stays empty."""
    sys.stderr.write("error: " + message + "\n")
    sys.stderr.write("usage: main.py lines|highlight FILE_A FILE_B\n")
    return 2


def main(argv):
    if len(argv) != 4 or argv[1] not in ("lines", "highlight"):
        return usage_error("wrong arguments")
    highlight = argv[1] == "highlight"
    try:
        # Both files are read before anything is computed or printed, so a
        # failure on the second file cannot leave half a diff on stdout.
        A = read_lines(argv[2])
        B = read_lines(argv[3])
    except OSError as exc:
        return usage_error("cannot read input: " + str(exc))
    cA, cB = diff_flags(A, B)
    out = build_output(A, B, cA, cB, highlight)
    sys.stdout.buffer.write(b"".join(out))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))