"""VisionLine optical-inspection module (corrected and optimised).

Functions
---------
sobel_edges(img, thr)            3x3 Sobel edge map (vectorised, no square root)
sobel_magnitude(img)             3x3 Sobel gradient magnitude (float array)
label_blobs(mask)                4-connected component labelling (single pass)
measure_parts(labels, n)         area, bounding box, perimeter, extent of ALL parts in one scan
match_template(img, tpl)         brute-force minimum-SSD search with early abandoning
Pixels outside the image are treated as 0 / "outside"; borders of the Sobel map are 0.
"""
import math
import numpy as np

try:                                    # OpenCV is optional: used only for cross-checks
    import cv2
except ImportError:                     # pragma: no cover
    cv2 = None


def sobel_magnitude(img):
    """Return the 3x3 Sobel gradient magnitude sqrt(gx^2 + gy^2); border pixels are 0."""
    a = np.asarray(img, dtype=np.int32)
    h, w = a.shape
    mag = np.zeros((h, w), dtype=np.float64)
    if h < 3 or w < 3:
        return mag
    gx, gy = _sobel_gradients(a)
    mag[1:-1, 1:-1] = np.sqrt(gx * gx + gy * gy)
    return mag


def _sobel_gradients(a):
    """Gx, Gy of the interior pixels, computed with nine shifted slices of the image."""
    tl, tc, tr = a[:-2, :-2], a[:-2, 1:-1], a[:-2, 2:]
    ml, mr = a[1:-1, :-2], a[1:-1, 2:]
    bl, bc, br = a[2:, :-2], a[2:, 1:-1], a[2:, 2:]
    gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl)      # kernel [-1 0 1; -2 0 2; -1 0 1]
    gy = (bl + 2 * bc + br) - (tl + 2 * tc + tr)      # kernel [-1 -2 -1; 0 0 0; 1 2 1]
    return gx, gy


def sobel_edges(img, thr):
    """Edge map: 1 where the Sobel magnitude is strictly greater than thr, else 0.

    sqrt is avoided by comparing gx^2 + gy^2 with thr^2 (valid because thr >= 0).
    The first/last row and column (no complete 3x3 window) are always 0.
    """
    a = np.asarray(img, dtype=np.int32)
    h, w = a.shape
    edge = np.zeros((h, w), dtype=np.uint8)
    if h < 3 or w < 3:
        return edge
    gx, gy = _sobel_gradients(a)
    edge[1:-1, 1:-1] = (gx * gx + gy * gy > thr * thr)
    return edge


def label_blobs(mask):
    """Label the 4-connected components of a binary mask (1 = part).

    Returns (labels, count); labels are numbered 1..count in raster order of each part's first pixel.
    Uses an explicit stack, so there is no recursion limit; every pixel is pushed at most once.
    """
    m = np.asarray(mask, dtype=np.uint8)
    h, w = m.shape
    lab = np.zeros((h, w), dtype=np.int32)
    ml, ll = m.tolist(), lab.tolist()                 # python lists are faster for scalar access
    count = 0
    for r in range(h):
        for c in range(w):
            if ml[r][c] == 1 and ll[r][c] == 0:       # a new, unlabelled seed pixel
                count += 1
                ll[r][c] = count
                stack = [(r, c)]
                while stack:
                    y, x = stack.pop()
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if 0 <= ny < h and 0 <= nx < w and ml[ny][nx] == 1 and ll[ny][nx] == 0:
                            ll[ny][nx] = count
                            stack.append((ny, nx))
    return np.array(ll, dtype=np.int32), count


def label_blobs_cv(mask):
    """Same partition as label_blobs, computed by OpenCV (4-connectivity)."""
    n, lab = cv2.connectedComponents(np.asarray(mask, dtype=np.uint8), connectivity=4)
    return lab.astype(np.int32), n - 1


def measure_parts(labels, n):
    """Measure all n parts in ONE pass over the label image.

    Returns {k: (area, (r0, c0, r1, c1), perimeter, extent, kind)}.
    perimeter = number of part pixels having a 4-neighbour outside the part (the image border counts as outside)
    extent    = area / bounding-box area, where the box has (r1 - r0 + 1) * (c1 - c0 + 1) pixels
    kind      = 'RECT' when extent >= 0.9, otherwise 'NON-RECT'.
    """
    lab = np.asarray(labels, dtype=np.int32)
    pad = np.pad(lab, 1)                                  # zero border = "outside"
    centre = pad[1:-1, 1:-1]
    boundary = (centre > 0) & ((pad[:-2, 1:-1] != centre) | (pad[2:, 1:-1] != centre) |
                               (pad[1:-1, :-2] != centre) | (pad[1:-1, 2:] != centre))
    rows, cols = np.nonzero(lab)
    ids = lab[rows, cols]
    area = np.bincount(ids, minlength=n + 1)
    perim = np.bincount(lab[boundary], minlength=n + 1)
    big = lab.shape[0] + lab.shape[1] + 1
    r0 = np.full(n + 1, big); c0 = np.full(n + 1, big); r1 = np.full(n + 1, -1); c1 = np.full(n + 1, -1)
    np.minimum.at(r0, ids, rows); np.minimum.at(c0, ids, cols)
    np.maximum.at(r1, ids, rows); np.maximum.at(c1, ids, cols)
    out = {}
    for k in range(1, n + 1):
        box = (int(r1[k] - r0[k] + 1)) * (int(c1[k] - c0[k] + 1))
        ext = area[k] / box
        out[k] = (int(area[k]), (int(r0[k]), int(c0[k]), int(r1[k]), int(c1[k])), int(perim[k]), float(ext),
                  'RECT' if ext >= 0.9 else 'NON-RECT')
    return out


def match_template(img, tpl, early_abandon=True, counter=None):
    """Brute-force minimum-SSD template search; returns (row, col, ssd) of the first best position.

    early_abandon: stop summing a window as soon as its partial SSD exceeds the best value so far
    (the result is identical because SSD terms are non-negative).
    counter: optional one-element list that receives the number of pixel comparisons performed.
    Returns None if the template is larger than the image.
    """
    a = np.asarray(img, dtype=np.int64).tolist()
    t = np.asarray(tpl, dtype=np.int64).tolist()
    H, W, h, w = len(a), len(a[0]), len(t), len(t[0])
    if h > H or w > W:
        return None
    best, pos, work = float('inf'), None, 0
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            ssd = 0
            for i in range(h):
                ai, ti = a[r + i], t[i]
                for j in range(w):
                    d = ai[c + j] - ti[j]
                    ssd += d * d
                    work += 1
                    if early_abandon and ssd > best:
                        break
                else:
                    continue
                break
            if ssd < best:
                best, pos = ssd, (r, c)
    if counter is not None:
        counter[0] = work
    return pos[0], pos[1], int(best)


def match_template_cv(img, tpl):
    """cv2.matchTemplate(TM_SQDIFF) version used to cross-check match_template."""
    res = cv2.matchTemplate(np.asarray(img, np.float32), np.asarray(tpl, np.float32), cv2.TM_SQDIFF)
    mn, _, loc, _ = cv2.minMaxLoc(res)
    return loc[1], loc[0], int(round(mn))


# ----------------------------------------------------------------------------------------------
# Test harness
# ----------------------------------------------------------------------------------------------
def _same_partition(a, b):
    """True if two label images describe the same set of parts (numbering may differ)."""
    pairs = set(zip(a.ravel().tolist(), b.ravel().tolist()))
    return len({p[0] for p in pairs}) == len(pairs) == len({p[1] for p in pairs})


def run_tests():
    img = [[10, 10, 10, 10, 10, 10], [10, 10, 10, 200, 200, 200], [10, 10, 10, 200, 200, 200],
           [10, 10, 10, 200, 200, 200], [10, 10, 10, 200, 200, 200]]
    mask = [[1, 1, 1, 0, 0, 0, 0, 1], [1, 1, 1, 0, 1, 1, 0, 1], [1, 1, 1, 0, 1, 1, 0, 0], [0] * 8,
            [0, 1, 0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 0, 0, 0, 0]]
    scene = [[3, 1, 4, 1, 5, 9], [2, 6, 5, 3, 5, 8], [9, 7, 9, 3, 2, 3], [8, 4, 6, 2, 6, 4], [3, 3, 8, 3, 2, 7]]
    tpl = [[6, 4], [2, 7]]
    # --- Sobel ---------------------------------------------------------------------------------
    e = sobel_edges(img, 100)
    assert e.tolist() == [[0] * 6, [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0], [0] * 6]
    assert e[0].sum() == 0 and e[-1].sum() == 0 and e[:, 0].sum() == 0 and e[:, -1].sum() == 0, 'no edges on the border'
    assert e[1:4, 2:4].all(), 'edge on the step'
    m = sobel_magnitude(img)
    assert abs(m[1][2] - 600.8) < 0.1 and abs(m[1][3] - 806.1) < 0.1 and abs(m[2][2] - 760.0) < 0.1
    assert np.array_equal(sobel_edges(img, 100), (m > 100).astype(np.uint8)), 'squared compare == sqrt compare'
    assert not sobel_edges([[7] * 5] * 5, 0).any(), 'uniform image has no edges'
    assert not sobel_edges([[1, 2], [3, 4]], 0).any(), 'image smaller than 3x3'
    if cv2 is not None:
        big = np.random.default_rng(1).integers(0, 256, (40, 50)).astype(np.uint8)
        cvm = np.sqrt(cv2.Sobel(big, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REFLECT_101) ** 2 +
                      cv2.Sobel(big, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REFLECT_101) ** 2)
        assert np.allclose(sobel_magnitude(big)[1:-1, 1:-1], cvm[1:-1, 1:-1]), 'agrees with cv2.Sobel'
    # --- labelling -----------------------------------------------------------------------------
    lab, n = label_blobs(mask)
    assert n == 5 and lab.tolist() == [[1, 1, 1, 0, 0, 0, 0, 2], [1, 1, 1, 0, 3, 3, 0, 2], [1, 1, 1, 0, 3, 3, 0, 0],
                                       [0] * 8, [0, 4, 0, 0, 5, 5, 5, 0], [0, 4, 4, 0, 0, 0, 0, 0]]
    assert label_blobs([[0, 0], [0, 0]])[1] == 0, 'empty mask'
    assert label_blobs([[1]])[1] == 1, '1 x 1 mask'
    assert label_blobs([[1, 0], [0, 1]])[1] == 2, 'diagonal pixels are different parts (4-connectivity)'
    if cv2 is not None:
        rnd = (np.random.default_rng(2).random((60, 70)) > 0.55).astype(np.uint8)
        a, na = label_blobs(rnd); b, nb = label_blobs_cv(rnd)
        assert na == nb and _same_partition(a, b), 'agrees with cv2.connectedComponents'
    # --- measurement ---------------------------------------------------------------------------
    s = measure_parts(lab, n)
    assert s[1] == (9, (0, 0, 2, 2), 8, 1.0, 'RECT')                      # touches the top and left border
    assert s[2] == (2, (0, 7, 1, 7), 2, 1.0, 'RECT')                      # one pixel wide, touches the right border
    assert s[3] == (4, (1, 4, 2, 5), 4, 1.0, 'RECT')
    assert s[4] == (3, (4, 1, 5, 2), 3, 0.75, 'NON-RECT')                 # chipped part (L-shape)
    assert s[5] == (3, (4, 4, 4, 6), 3, 1.0, 'RECT')                      # one pixel high
    full = measure_parts(np.ones((4, 5), np.int32), 1)[1]
    assert full == (20, (0, 0, 3, 4), 14, 1.0, 'RECT'), 'whole image is one part (border counts as outside)'
    assert measure_parts(np.array([[1]]), 1)[1] == (1, (0, 0, 0, 0), 1, 1.0, 'RECT'), 'single pixel'
    # --- template matching ---------------------------------------------------------------------
    assert match_template(scene, tpl) == (3, 4, 0)
    assert match_template(scene, tpl, early_abandon=False) == (3, 4, 0)
    last = [[1, 2], [3, 4], [9, 9]]
    assert match_template(last, [[9, 9]]) == (2, 0, 0), 'template in the last row is found'
    assert match_template(scene, scene) == (0, 0, 0), 'template equal to the whole image'
    assert match_template([[1, 2]], [[1, 2], [3, 4]]) is None, 'template larger than image'
    if cv2 is not None:
        rng = np.random.default_rng(3)
        im = rng.integers(0, 256, (30, 40)); tp = im[11:17, 9:16].copy()
        assert match_template(im, tp) == match_template_cv(im, tp) == (11, 9, 0), 'agrees with cv2.matchTemplate'
    return 'ALL TESTS PASSED'


if __name__ == '__main__':
    print(run_tests())
