# Changelog

Every published version, newest first.  This file is on the publish
allow-list, so it travels with the package: it is the only thing a
consumer deciding whether to upgrade can read.

## 0.0.1 — 2026-09-11

The **interface**, before anyone implements it.  Every signature, every
type and every effect row is published; every body is `todo()`, and the
release is stamped `NOT IMPLEMENTED — interface only`.  Adding this
package works and calling it panics.

- Nine modules and no dependencies.  `ggufread` is the index and the
  reader, `ggufquant` the GGML block layouts, `ggufinfo` the tensor
  table, `ggufmeta` the metadata table, `ggufval` the thirteen value
  types, `ggufhdr` the header and the alignment, `ggufrange` the byte
  range, `ggufwrite` the writer and `gguferr` the fault.
- **The reader asks and the host performs.**  Nothing in GGUF records
  where the metadata table ends, so no caller can be told in advance
  how many bytes the index is.  `ggufread.need` answers a byte range
  and `ggufread.offer` takes it back, four functions and a six-line
  host loop — an mmap, a file descriptor, an HTTP range request or a
  device reading flash, all the same shape.  A 40 GB checkpoint goes
  through it in a few hundred kilobytes of asks.
- **The block layouts are the reason the package exists.**  Every GGML
  type gives its elements-per-block and its bytes-per-block: F32
  through Q8_1, the K-quants, the nine IQ types and the two ternary
  ones, thirty-one in all, with the five withdrawn codes named rather
  than reported as unknown.  A Q4_K tensor of 4096 by 4096 is
  9,437,184 bytes, and nothing else on the grid can tell a caller that.
- **Dequantisation is deliberately elsewhere.**  It lives in `ml`,
  which keeps the kernels and the device dispatch; this package owns
  the layout so both agree on one spelling.  ml's own Q4_K-style
  supergroup is 168 bytes over 256 weights and GGML's `block_q4_K` is
  144 over the same 256 — two different formats with similar names,
  and keeping the file format's numbers in the file format's package
  is what keeps that visible.
- **The byte order comes from the version field, not the magic.**  The
  four characters are `GGUF` in file order in both byte orders; a
  version whose low sixteen bits are zero was read backwards.  The
  reference implementation's own test, copied.
- **A tensor's offset is relative and its range is absolute**, and the
  package keeps both and names which is which — a reader that mixed
  them has a first tensor that is right and a last one that is
  megabytes wrong.
- **`general.file_type` is a hint**; the tensor-info table is the
  authority, and `ggufinfo.type_histogram` is what says a "Q4_K_M"
  file is four quantisations.
- The writer emits the **index only** — header, both tables, padding —
  and never a weight, so changing a chat template on a 40 GB
  checkpoint decodes nothing.
- No `@tier(embedded)` claim, and the README says why rather than
  leaving it as an omission.
- `tests/` holds twenty-three API tests across two files, every one
  red.  The block sizes are asserted as the field sums they are —
  `2 + 2 + 12 + 128 = 144` — so a wrong one is visible as a wrong
  addition; `gguf-py` over real checkpoints becomes a generated run
  beside them when the bodies land.
