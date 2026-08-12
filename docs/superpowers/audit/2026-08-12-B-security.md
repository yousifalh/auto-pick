# Security audit — B

**Scope:** read-only security review of `auto-pick` ahead of public release.
**HEAD:** `fa7a4f0`. **Date:** 2026-08-12. **Threat model:** public portfolio
repository; no production deployment, no real robot attached. Severities are
rated for *that* model, not for a hypothetical factory install.

Findings marked **[verified]** were confirmed by executing code. Findings marked
**[read]** were established by reading the source only.

---

## Summary

| # | Severity | Finding | File |
|---|----------|---------|------|
| 1 | **High** | `torch.load` without `weights_only=True` on the detector path → arbitrary code execution from a hostile checkpoint | `recog/inference.py:84` |
| 2 | **Medium** | Client crashes on an out-of-range coordinate **without firing the E-stop** — the only failure path that skips it | `execution/execution.py:193` |
| 3 | **Low** | One valid 16-byte packet makes the mock server sleep up to 63 days; no workspace envelope check | `execution/mock_kuka_server.py:67` |
| 4 | **Low** | Mock server has no socket timeout and no connection cap; half-open connections pin threads indefinitely | `execution/mock_kuka_server.py:177` |
| 5 | **Low** | Dependency floors admit vulnerable Pillow; `opencv-python` pulls a GUI stack | `pyproject.toml:9-10` |
| 6 | **Informational** | Absolute/relative `file_name` in a COCO annotation escapes `img_dir` | `recog/dataset.py:395`, `recog/seg_dataset.py:320` |

Clean categories, stated plainly: **no** `pickle`, **no** `eval`/`exec`, **no**
`subprocess` or `os.system` anywhere in the tree; every YAML load is
`yaml.safe_load`; no `np.load`/`allow_pickle`; no dynamic imports driven by
config values; no secrets in the working tree or in 177 commits of history.

---

## 1. `torch.load` without `weights_only=True` — High

**`recog/inference.py:84`**

```python
state = torch.load(checkpoint_path, map_location="cpu")
```

`torch.load`'s default (`weights_only=False` on torch < 2.6) runs the pickle
machinery, so a checkpoint file can execute arbitrary code at load time via
`__reduce__`. No CRC, signature or provenance check precedes it.

**What an attacker achieves.** Anyone who can hand the user a `.pt` file gets
code execution as that user. This is not theoretical for this repo
specifically:

- No checkpoints are committed (`git ls-files` finds no `.pt`/`.pth`/`.ckpt`),
  so every user who wants to reproduce the README's results **must obtain a
  checkpoint from somewhere else** — a release asset, the author, or a fork.
- `README.md:269,284` document `--checkpoint recog/checkpoints/best.pt` as the
  headline commands. Following the README is the vulnerable act.
- `recog/inference.py:350` (`build_detector`) is the sole construction site, and
  it is on the main end-to-end path (`main.py` → detector → planner → robot).

**The project already knows the rule**, which is what makes this a miss rather
than an oversight: `recog/bay_segmenter.py:81-86` carries the comment
*"weights_only=True is not optional. The default unpickles arbitrary Python
objects, so loading a checkpoint becomes arbitrary code execution by whoever
produced the file."* The rule was written down and then not applied one module
over — in the loader for the *other* model, on the more prominent path.

**Fix (one line):** add `weights_only=True` to the call at
`recog/inference.py:84`. The surrounding code already handles the resulting
object correctly (`state["model"] if "model" in state else state` works on a
plain tensor dict).

### Full `torch.load` call-site table

| File:line | `weights_only` | Verdict | Note |
|---|---|---|---|
| `recog/inference.py:84` | **absent** | **UNSAFE** | Detector on the main path. The finding. |
| `recog/bay_segmenter.py:85` | `True` | Safe | Correct, with the explanatory comment. |
| `recog/seg_evaluate.py:619` | `True` | Safe | |
| `recog/seg_evaluate.py:670` | `True` | Safe | |
| `recog/seg_training.py:458` | `False` **(explicit)** | **Acceptable** | Justified in-line: `train_state.pt` carries optimiser/scheduler state that is not tensors-only, and is written and read by the same script on the same machine. It is reached only behind `--resume`, which additionally requires the file to already exist locally (`:449`). Not a distribution vector. Leave as is. |

That is every `torch.load` in the repository — four in `recog/`, one in
`seg_training.py`; remaining grep hits are prose in `docs/` and a code block in
a plan document. **[verified]** by grep across all `*.py` and by reading each
site.

---

## 2. Out-of-range coordinate bypasses the E-stop — Medium

**`execution/execution.py:193`**, inside `_cmd_and_wait`:

```python
try:
    self._send(pack_command(op, x, y, z, aux))
    return self._recv_status()
except (socket.timeout, ValueError) as exc:
    ...
```

`pack_command` packs `x`/`y` as `>i` (int32) and `z` as `h` (int16) with **no
clamping** (`execution/protocol.py:96-102`). A coordinate outside those ranges
raises `struct.error`.

**[verified] `struct.error` is not a subclass of `ValueError`** — confirmed by
executing `issubclass(struct.error, ValueError)` → `False`, and by calling
`pack_command(OpCode.MOVE_TO, x_mm=2**31)` → `struct.error`, likewise
`z_mm=40000` → `struct.error`.

**What this achieves.** The `except` clause does not catch it, so the exception
propagates out of `_cmd_and_wait` uncaught. Every *other* failure mode in this
client — CRC error, socket timeout, retry exhaustion — funnels into
`self.estop()` at `:206` before raising, which the module docstring ties to the
"Category-0 immediate-stop requirement in PPR §7.3 (R4)". An out-of-envelope
coordinate is the single path that tears down the call stack with **the vacuum
possibly still on and the arm mid-motion, and no E-stop sent**. The safety
invariant the class is built around has exactly one hole, and it is the one
reachable from a bad perception result: any planner or calibration bug that
produces a pose beyond ±2.1 m (x/y) or ±32.7 m (z) hits it.

Note the asymmetry within `pack_command` itself: `aux` *is* masked
(`int(aux) & 0xFFFF`, **[verified]**: `aux=0x1FFFF` round-trips to `65535`),
but the three coordinates are not. The masking habit was applied to the one
field where silent truncation is harmless and omitted from the three where a
range violation matters.

**Fix (one line):** widen the handler to
`except (socket.timeout, ValueError, struct.error) as exc:` — or, better for
the deployment story, validate the pose against a configured workspace envelope
in `move_to`/`pick_and_place` before packing.

---

## 3 & 4. Mock server robustness — Low (each)

### The parsing itself is sound

`execution/protocol.py` resists everything thrown at it. **[verified]** by
executing a fuzz harness:

- **200,000 random inputs** at lengths 0/1/15/16/17/64 through both
  `unpack_command` and `unpack_status`: **zero** exceptions escaped that were
  not `ValueError`. The handler at `mock_kuka_server.py:132` catches exactly
  `ValueError`, so the contract holds and every malformed packet becomes a
  `CRC_ERROR` status rather than a dead thread.
- **All 256 opcode bytes** with a *valid* CRC: unknown opcodes raise
  `ValueError` via `OpCode(op_byte)` (`protocol.py:125`) — no non-`ValueError`
  escape.
- Length is checked **exactly** (`!= COMMAND_LEN`), not as a lower bound, and
  CRC is verified *before* any field is interpreted, then version, then opcode.
  That is the right order.
- **[verified]** a bad-CRC packet against a live server returns status code
  `0x05` (`_CRC_ERROR`) and keeps the connection open.

There are no unchecked lengths, no buffer assumptions and no integer-overflow
handling errors in the parser. `struct.unpack` on a fixed 14-byte format with a
pre-validated length cannot over-read. **Verdict: the protocol parsing is
clean.** The two issues below are in the server's *I/O and simulation* layers,
not its parser.

### 3. Unbounded sleep from a single valid packet — `mock_kuka_server.py:67`

```python
t_ms = int(dist / 100.0 * self.ms_per_100mm + 50)
time.sleep(t_ms / 1000)
```

`dist` is the Euclidean distance to attacker-supplied coordinates, with no
workspace clamp anywhere in `_RobotState.move_to`. **[verified]** by
computation: a single well-formed, correct-CRC `MOVE_TO(2^31-1, 2^31-1, 32767)`
yields `time.sleep(5_466_600 s)` = **63.3 days**, holding the handler thread
and the simulated robot in a wedged state.

For the mock this is a self-DoS curiosity. It is listed because it is the same
missing check as finding 2 seen from the other end of the wire: **neither side
validates that a commanded pose lies inside the robot's envelope.** For a
system whose README is about driving a KR 6 R700, that absence is worth naming
even though no hardware is attached.

**Fix (one line):** clamp or reject coordinates outside the configured envelope
at the top of `move_to`.

### 4. No socket timeout, no connection cap — `mock_kuka_server.py:177`

`_recv_n` loops on `recv` until it has all 16 bytes. No `settimeout` is set on
the accepted socket anywhere in the module.

**[verified]** against a live server on port 54711:

- A client that sends **4 of 16 bytes** and stops gets no reply and no
  disconnect; the handler blocks in `_recv_n` indefinitely (2 s observation
  window, thread still held).
- **200 half-open connections** each sending one byte were accepted and held
  simultaneously — process thread count rose to 203. `_MockServer` is a
  `ThreadingTCPServer` with `daemon_threads = True` and no `request_queue_size`
  limit or concurrency cap.
- The server did keep serving new clients throughout (a fresh handshake
  succeeded while 200 connections were stuck), so this degrades rather than
  halts.

Contrast the client, which *does* set timeouts (`execution.py:100,218`). The
asymmetry is the tell: the timeout discipline was applied on one side only.

Realistic severity is **Low** — the default bind is `127.0.0.1`
(`mock_kuka_server.py:207,238`) and this is a test fixture. It rises if anyone
runs `python -m execution.mock_kuka_server --host 0.0.0.0`, which the CLI
permits without warning.

**Fix (one line):** `self.request.settimeout(...)` at the top of
`_Handler.handle`.

**Test coverage gap.** `execution/mock_kuka_server.py` has **no test file** —
`tests/` contains `test_protocol.py` and `test_execution.py` but no
`test_mock_kuka_server.py`. The 13 tests in `test_protocol.py` cover CRC
vectors, round-trips, wrong length, CRC corruption and bad version — good
coverage of the *parser*, which is why the parser is clean. Nothing exercises
the *server's* framing, truncation or concurrency behaviour, which is where
findings 3 and 4 live. That correlation is not a coincidence.

---

## 5. Dependencies — Low

**`pyproject.toml:7-25`.** These are **floors** (`>=`), not pins, and there is no
lockfile. A fresh install resolves to current releases — **[verified]** in this
environment: numpy 2.4.3, pillow 12.2.0, pyyaml 6.0.3, albumentations 2.0.8,
opencv-python 5.0.0.93, torch 2.13.0. All current and unaffected.

The exposure is that a *reproducible* or constrained install may legitimately
resolve at the floor:

| Package | Floor | At the floor |
|---|---|---|
| `pillow>=10.0` | 10.0 | **CVE-2023-4863** (libwebp heap buffer overflow, critical, fixed 10.0.1) and **CVE-2023-50447** (arbitrary code execution via `PIL.ImageMath.eval`, fixed 10.2.0). Pillow decodes attacker-supplied images throughout `recog/`. Raise the floor to `pillow>=10.3`. |
| `numpy>=1.24` | 1.24 | No serious advisory. Fine. |
| `pyyaml>=6.0` | 6.0 | Fine — the `yaml.load` RCE affects <5.4, and all four call sites use `safe_load` regardless. |
| `opencv-python>=4.8` | 4.8 | See below. |
| `albumentations>=1.4` | 1.4 | No serious advisory. |
| `torch>=2.0` (extra) | 2.0 | Predates `weights_only=True` becoming the default in 2.6 — which is precisely why finding 1 must be fixed explicitly rather than left to the torch default. |

**The one I would not ship as written: `opencv-python`.** No source file uses
`cv2.imshow`, `waitKey`, `namedWindow`, `destroyAllWindows`, `createTrackbar`
or `setMouseCallback` — **[verified]** by grep across the tree, zero hits. The
GUI build is therefore pulling a Qt/GTK/X11 stack (`libGL`, `libglib`) that is
never called, adding install weight, a class of container failures
(`ImportError: libGL.so.1`) and needless attack surface in CI. Swap to
`opencv-python-headless>=4.8`; it is a drop-in here.

Combined fix: `pillow>=10.3` and `opencv-python-headless>=4.8`.

---

## 6. Path handling — Informational

**[verified]** by executing the join semantics:

- `Path("recog/dataset") / "/etc/passwd"` → `\etc\passwd` — an **absolute**
  `file_name` discards the base directory entirely (pathlib's documented `/`
  behaviour).
- `os.path.join("recog/dataset", r"C:\Windows\win.ini")` → `C:\Windows\win.ini`
  — same escape on the `os.path` side.
- Relative traversal (`../../../..`) is preserved unresolved.

Reached at:

- **`recog/dataset.py:395`** — `self._load_image(self.img_dir / rec.file_name)`,
  where `file_name` comes straight from the COCO JSON at `:209`.
- **`recog/seg_dataset.py:320`** — `os.path.join(self.img_dir,
  img_meta["file_name"])`, likewise unvalidated.

**What a malformed input achieves:** a crafted annotation JSON makes the loader
*read* files outside the dataset directory. It is a read-only primitive whose
output is decoded as an image into a training tensor — there is no write, no
execution, and no channel back to the attacker. Anyone who can supply your
COCO file has already persuaded you to run their training config, which is a
strictly larger capability. **Not a real finding for this project**, recorded
only for completeness.

Worth noting the counter-example, because it shows the safe pattern is already
known here: **`recog/labelme_to_seg.py:205`** does
`file_name = Path(image_path_field).name`, which strips any directory component
and neutralises traversal from the untrusted `imagePath` field. That file is
clean. So is `recog/check_annotations.py`. If finding 6 is ever addressed, the
fix is to apply `labelme_to_seg.py`'s `.name` treatment in the two loaders
above.

Config values reaching the filesystem (`common/config.py:52-54`, checkpoint
paths in `main.py:236-257`) are CLI-supplied paths in a developer tool —
resolving them relative to the project root is the intended behaviour, not a
traversal bug. `common/config.py` uses `yaml.safe_load` and validates existence
before opening.

---

## 7. Secrets in tree and history — clean

**[verified]** across the working tree and all **177 commits**:

- No API keys, tokens, passwords, PEM private-key headers, AWS/GitHub/Slack
  credential patterns. The handful of regex hits are the English word "token"
  in prose and one assertion message (`tests/test_synth3d.py:264`).
- `git ls-files` tracks no `.env`, `.pem`, `.key`, `.p12`, `id_rsa` or
  credential file, and no such file appears among deleted paths in history.
- **The student-ID redaction took.** Searching all history for standalone 6–9
  digit identifiers returns only date-derived RNG seeds (`MASTER_SEED =
  20260806`, `default_rng(20260811)`). The surviving references are the
  placeholder `Student ID: *[fill in]*` in `docs/FDR_v2.md:6` and the audit
  notes describing the redaction itself. No real ID remains.
- No checkpoints, datasets or binary blobs are tracked that would carry
  embedded metadata.

One item to decide on rather than fix: git history carries the author's
personal email (`yousifalh26@icloud.com`) on all commits. For a personal
portfolio repo that is normal and probably intended; flagging only so the
choice is deliberate before the repo goes public.

---

## Recommended actions, in order

1. **`recog/inference.py:84`** — add `weights_only=True`. Do this before
   publishing; it is the finding that touches people who download a checkpoint.
2. **`execution/execution.py:195`** — add `struct.error` to the caught
   exceptions so the E-stop path cannot be bypassed, or validate the pose
   envelope before packing.
3. **`pyproject.toml`** — `pillow>=10.3`, and `opencv-python-headless>=4.8` in
   place of `opencv-python`.
4. **`execution/mock_kuka_server.py`** — a socket timeout in `_Handler.handle`
   and a coordinate clamp in `move_to`; add a `tests/test_mock_kuka_server.py`
   covering truncated frames and unknown opcodes.

Items 1 and 3 are one-line changes. Items 2 and 4 are small and improve the
robotics-safety story a reader of this repo is being asked to evaluate.
