# gguf-nv

**Status: NOT IMPLEMENTED — interface only.**

Every public function below is published with its signature and its
effect row, and every body is `todo()`.  Installing this package works;
calling it panics with `not implemented`.

## What this is

A **GGUF model file, read with no file in it**.  The header, the
metadata key/value table, the tensor-info table, and — the part nothing
else gives you — the GGML block layouts as numbers, so that a caller can
work out exactly which bytes of a 40 GB checkpoint one tensor occupies
and read only those.

Nine modules, and no dependencies.

| surface | module | reach for it when |
| --- | --- | --- |
| the **index** | `ggufread` | you have a file and want to know what is in it |
| the **block layouts** | `ggufquant` | you are sizing, slicing or reporting a quantised tensor |
| the **tensor table** | `ggufinfo` | you want one tensor's bytes |
| the **metadata** | `ggufmeta` | you want a hyperparameter, a vocabulary or a chat template |
| the **values** | `ggufval` | you are reading or building a metadata value by hand |
| the **header** | `ggufhdr` | you are sniffing a file, or aligning something |
| the **place** | `ggufrange` | you are doing your own reads |
| the **writer** | `ggufwrite` | you are putting a file back together |
| the **faults** | `gguferr` | you are telling somebody why their model did not load |

## Adding it, and checking it

```bash
novo pkg add gguf-nv              # into your novo.toml
novo pkg build                    # type- and effect-check the package
novo test --isolate tests/ggufquant_tests.nv
```

`novo test` is red today and that is the point of the release: every
assertion fails with `not implemented: gguf-nv.<module>.<fn>`.  They
turn green one at a time as bodies land.

## The one example that will work

```novo
use std.bytes
use ggufread

fn main() [io]
    match ggufread.parse_index(bytes.zeros(0))
        Err(f)  => println(f.message())
                   // : not a GGUF file: first four bytes read 0
        Ok(ix)  => println("${ix.header.tensor_count} tensor(s)")
```

## The load-bearing interface

Two of them, and the first is what makes the package possible at all.

### `ggufread.need` and `ggufread.offer` — the core asks, the host performs

```novo norun:pseudo
var r = ggufread.reader()
loop
    if ggufread.is_done(r)
        break
    let want = ggufread.need(r)             // a GgufRange: at, len
    r = ggufread.offer(r, host_read(want))!
let ix = ggufread.index_of(r)!
```

A `core` package's budget is no effects at all, so it may not open a
file, seek in one or map one.  A GGUF's index also cannot be read in one
go: **nothing in the format records where the metadata table ends.**  A
reader learns that only by walking it — key, type tag, value, repeat,
`metadata_kv_count` times — and one of those values may be a
128,256-element array of strings.  So neither "hand me the whole file"
nor "hand me the first N bytes" is a contract a caller can keep in
advance.

So the reader **asks**.  Four functions, and the host's loop is the six
lines above whether the host is an mmap, a file descriptor, an HTTP
range request against a model repository, or a device reading external
flash.  `docs/publishing.md` § How a `core` package takes bytes from its
host calls this "the core asks, the host performs" and lists it as the
right shape for an indexed format that streaming would throw away.

The reader never holds the file.  A 40 GB checkpoint goes through it in
a few hundred kilobytes of asks, because the only bytes it ever sees are
the index.

`ggufread.read_index` is the same machine with the pump written once,
for the caller whose source is sequential:

```novo norun:pseudo
pub fn read_index<S: Read[e]>(src: S) -> Result<GgufIndex, GgufFault> [e]
```

The bound binds `Read`'s effect parameter and the clause uses it
(SPEC § 5.6), so the row means "whatever the impl behind `S` supplies" —
`[io]` for a file, `[io, net]` for a socket, nothing for an in-memory
buffer — and the package stays `core`.  It is the only effect row in
gguf-nv, and it spends nothing.

### `ggufquant.block_elements` and `ggufquant.block_bytes` — two numbers per type

```novo norun:pseudo
pub fn block_elements(t: GgufType) -> Int      // 1, 32 or 256
pub fn block_bytes(t: GgufType) -> Int         // Q4_K: 144
```

This is what the package exists for.  Everything else here reads a
table; this is the part a caller cannot derive, cannot guess, and gets
wrong in a way that produces a model which loads and speaks nonsense.

A Q4_K tensor of 4096 by 4096 is not 8 MB and it is not 12 MB.  It is
4096 rows of 16 blocks of 144 bytes — **9,437,184** — and a loader that
computed it any other way reads the next tensor's bytes as the end of
this one.  Everything above it follows: `row_bytes`, `tensor_bytes`,
`block_range`, and every `GgufRange` in the package.

Two block sizes cover every quantised type.  The original Q4_0 family
works **32** weights at a time with one scale each; the K-quants and the
IQ family work **256** at a time with a scale hierarchy — a
per-superblock scale and sixteen or eight sub-block ones packed six bits
each.  That is why a row length must be a multiple of the block size,
and why a Q4_K tensor whose first dimension is 100 does not exist.

## Dequantisation is not here, and that is the split with `ml`

`orbit/ml` is the native inference engine this row was cut from, and the
split is clean because the two halves never needed each other's code:

| | |
| --- | --- |
| **ml keeps** | the dequantisation kernels — the CUDA and ROCm mat-vecs, the upload-time quantiser that turns BF16 weights into blocks — and the device dispatch that chooses between CPU, CUDA and ROCm |
| **ml would take from gguf-nv** | the header and both tables, the type codes, and the two block numbers |

Today ml reads **safetensors**, not GGUF: `ml/src/ml/loader.nv` walks a
HuggingFace checkpoint directory, and `ml/src/ml/gate.nv` refuses a
pre-quantised repository by name and tells the user to fetch the
unquantised one.  Taking gguf-nv is what would let it open the files
most people actually have, and it needs nothing from this package but
the index and the block sizes — it already owns every kernel.

**The number that makes the split worth stating out loud**: ml's own
Q4_K-*style* supergroup is **168 bytes** over 256 weights (8 + 32 + 128,
in `ml/backend/common/mlkern.cu`), and GGML's `block_q4_K` is **144**
over the same 256.  They are different formats with similar names.  The
only way that stays visible is for the file format's numbers to live in
the file format's package, with the file format's spelling on them —
which is this package, and which is why `ggufquant` publishes
`is_file_type`, `withdrawn_name` and a type histogram rather than a
single "bytes per weight" that would blur the two.

## Every answer about a position is a range

```novo norun:pseudo
pub struct GgufRange
    at: Int       // from byte 0 of the FILE
    len: Int
```

To a caller that mapped the whole file it is a **span**, and
`ggufrange.slice` cuts it out.  To a caller that mapped none of it — a
loader streaming a checkpoint into VRAM a tensor at a time — it is a
**request**: read these bytes at this offset and hand them back.  One
type for both, because they are one fact.

elf-nv's `ElfRange` is the house style and this is deliberately the same
shape, so a reader who has met one meets no surprise in the other.  What
this package does not do is depend on it: struct identity is keyed by
name across an assembly, the two formats share nothing but the idea, and
a model loader downloading an ELF reader to get a pair of integers would
be a dependency nobody asked for.

**The offset on a tensor entry is relative and the range is absolute.**
The file stores an offset counted from the start of the data blob, and
`GgufTensorInfo.offset` keeps it that way so that a round trip through
`ggufwrite` writes back the number that was read.  Every function that
answers a *position* takes the blob's own start and returns a range
counted from byte 0.  A reader that mixed them is a reader whose first
tensor is correct and whose last one is megabytes wrong.

## Three things about GGUF that surprise everyone

**The magic does not tell you the byte order.**  The four bytes are
`GGUF` in file order in a little-endian file and in a big-endian one
alike, because they are written as characters rather than as a number.
The **version** field decides it: a version whose low sixteen bits are
zero was read backwards, because no real version is a multiple of 65536.
That is `ggufhdr.detect_order`, and it is the reference implementation's
own test.

**The architecture prefixes its own keys.**  A Llama model's layer count
is `llama.block_count` and a Qwen2 model's is `qwen2.block_count`, so a
reader has to read `general.architecture` first and build every other
key name from it.  `ggufmeta.arch_key` is that concatenation and the
eight named helpers over it are the hyperparameters every decoder-only
architecture declares.

**`general.file_type` is a hint, not a fact.**  It names what the
quantiser was *asked* for; the tensor-info table names what each tensor
actually *is*, and a mixed quantisation makes them disagree by design.
`ggufinfo.type_histogram` is what tells a user their "Q4_K_M" file is
four different quantisations, which it is.

## The layer, and the claim this package does not make

`core`, and the format makes it easy: a GGUF is an index, and reading an
index is arithmetic over bytes the caller already holds.  There is not
one effect row in the package except `read_index`'s bound `[e]`, which
spends nothing.

**No `@tier(embedded)` claim**, and the reason is worth writing down
rather than leaving as an omission.  A device probe would build, because
nothing here needs a heap that a `Bytes` does not — but the claim would
be dishonest in a more useful sense: a GGUF is addressed by 64-bit
offsets into a file measured in gigabytes, the consumer is a host loader
feeding an accelerator, and the microcontroller that wants to read one
does not exist yet.  The audit's `core-embedded` row passes and says the
package makes no claim.  If an edge consumer appears — a device
streaming a 50 MB model off external flash — the split to make is
lz4-nv's: an integer-only module for the arithmetic and the `Bytes`
modules beside it.

## What this package does not depend on

**Not ndarray-nv.**  This package never holds a tensor.  It holds the
tensor's name, its dimensions, its GGML type and the byte range its data
occupies, and hands all four to whoever is going to dequantise.  An
array library in the dependency list would be downloaded by every
consumer and used by no function.

**Not a float16 package.**  The scalar widths reported here are integers,
and the one place a half-precision *number* would be needed is
dequantisation, which is not here.  No GGUF metadata value is ever F16:
the thirteen value types are in `ggufval`, and F32 and F64 are the only
floats among them.

## Naming

Every public type and every enum variant starts `Gguf`, and every module
file starts `gguf`.  That is not decoration: struct and enum identity is
keyed by **name** across a whole assembly, dependencies included, so two
packages that both declare `Header` cannot be used by one program.  The
rule covers variant names, which is why the faults are `GgufBadMagic`
and `GgufOutputFull` rather than the short nouns.

It bites inside a package too.  The metadata value types are
`GgufTypeUint32` and the tensor types are `GgufQ4_K` — two enums that
would both want to be called "type", so one of them says so in every
variant.

## What is the specification, and what is this package's choice

**The GGUF specification and `ggml-common.h`, and binding**: the four
characters, the version field, the two counts and their two widths, the
thirteen value type codes, the string-as-length-and-bytes encoding, the
array's element type before its count, the tensor entry's
name/rank/dims/type/offset order, the maximum rank of four, the default
alignment of 32, the rule that the data blob starts at the first
multiple of the alignment after the tensor-info table, every GGML type
code, and every block struct's size.

**This package's choice**: refusing an unknown value-type code rather
than treating it as the end of the table; naming the five withdrawn
tensor types instead of reporting them as unknown; bounding array
nesting at eight, which the format does not bound at all; answering the
FIRST entry for a duplicated key, and offering `duplicates` rather than
refusing the file; refusing a misaligned tensor offset, which llama.cpp
reads and mis-addresses; `check_layout` being a separate call from
parsing, so a tool repairing a damaged checkpoint can read the table
before it is told the table is wrong; and the writer emitting the index
only, never the weights.

## The reference implementation

[ggml](https://github.com/ggml-org/ggml)'s GGUF specification, its
`gguf.c` reader and `ggml-common.h`'s block definitions, and
[gguf-py](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py)'s
`GGUFReader`, whose byte-order test this package copies.  The oracle
arrives with the bodies: a set of small published checkpoints read with
`gguf-py` and with this package, every key, every tensor offset and
every size compared, as a generated run beside the two suites in
`tests/`.

Apache-2.0.
