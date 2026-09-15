# gguf-nv

GGUF is a file format for storing models for inference with GGML and
executors based on it. It is a single file that carries a model's weights
and everything needed to load them, and it is the format llama.cpp and
Ollama distribute. The format is
[specified by the ggml project](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
This package reads a GGUF's index in novo-lang, with no file in it: it
answers what is in the file and where each tensor's bytes are, and the
caller does the reading.

**Status: NOT IMPLEMENTED — interface only.** Every function is declared
with its full signature, but every body is a `todo()` that panics when
called. The package is published so its design can be reviewed and
depended on before it is implemented. Version 0.1.0 will be the first
working release.

## What a GGUF is

A GGUF has four parts in this order. The **header** is the four characters
`GGUF`, a version number, and how many entries each of the two tables that
follow holds. The **metadata table** is a list of key/value pairs: the
architecture, the hyperparameters, the vocabulary, the chat template. The
**tensor-info table** is one entry per tensor, giving its name, its
dimensions, its type and where its data begins. The **data blob** is every
tensor's bytes, one after another.

The header and the two tables together are the **index**. On a 40 GB
checkpoint the index is a few hundred kilobytes. Everything this package
answers comes out of it.

A tensor's numbers are usually **quantised**: stored in fewer bits than a
float, in fixed-size groups called **blocks**. A block holds a fixed number
of weights and occupies a fixed number of bytes, and the two numbers are
what decide where a tensor ends. The original family works 32 weights at a
time with one scale for the group. The **K-quants** and the **IQ** family
work 256 weights at a time with a hierarchy of scales: one for the
superblock and sixteen or eight for the sub-blocks, packed six bits each.
A row's length must be a multiple of the block's weight count.

The **alignment** is the number the data blob's start is rounded up to, and
it is `general.alignment` in the metadata or 32 when that key is absent.
Every tensor's offset in the tensor-info table is counted from the start of
the data blob, not from the start of the file.

Every answer this package gives about where something is, is a
**`GgufRange`**: a byte offset counted from byte 0 of the file, and a byte
count. To a caller that mapped the whole file it is a span, and
`ggufrange.slice` cuts it out. To a caller that mapped none of it, it is a
request: read these bytes at this offset.

This package performs no input or output. It opens no file, seeks in none
and maps none. The one exception is `ggufread.read_index`, which takes a
source the caller supplies and is charged whatever that source costs.

| Quantity | Value |
| --- | --- |
| Magic, at offset 0 | the four characters `GGUF` |
| Versions this package reads | 1, 2, 3 |
| Header size, version 1 | 16 bytes |
| Header size, versions 2 and 3 | 24 bytes |
| Count and length width, version 1 | 4 bytes |
| Count and length width, versions 2 and 3 | 8 bytes |
| Default alignment | 32 bytes |
| Metadata value types | 13 |
| Maximum tensor rank | 4 |
| Array nesting this package allows | 8 levels |

Two examples of the block arithmetic, from `ggml-common.h`:

| Type | Weights per block | Bytes per block | The sum |
| --- | --- | --- | --- |
| `GgufQ4_0` | 32 | 18 | 2 + 16 |
| `GgufQ4_K` | 256 | 144 | 2 + 2 + 12 + 128 |

A Q4_K tensor of 4096 by 4096 is 4096 rows of 16 blocks of 144 bytes, which
is 9,437,184 bytes. It is neither 8 MB nor 12 MB, and a loader that
computed it either of those ways reads the next tensor's bytes as the end
of this one.

## Install

```
novo pkg add gguf-nv
```

## Example

```novo
use std.bytes
use ggufread
use ggufmeta

fn main() [io]
    // The whole file in memory. A caller that cannot hold a 40 GB
    // checkpoint uses `reader`, `need` and `offer` instead.
    let image = bytes.zeros(0)

    match ggufread.parse_index(image)
        Err(f) => println(f.message())
        Ok(ix) =>
            // What the file says it is. The architecture name prefixes
            // every hyperparameter key in the metadata table.
            match ggufmeta.architecture(ix.meta)
                Err(f)   => println(f.message())
                Ok(arch) => println("${arch}: ${ix.header.tensor_count} tensor(s)")

            // Where one tensor's bytes are, counted from byte 0 of the
            // file. Reading them is the caller's work.
            match ggufread.tensor_range(ix, "token_embd.weight")
                Err(f) => println(f.message())
                Ok(r)  => println("${r.len} bytes at ${r.at}")
```

Build and test with `novo pkg build` and `novo test`. Today `novo test`
fails on purpose: every test reaches a `not implemented: gguf-nv.<module>.<fn>`
panic. The tests are the specification the implementation will have to
satisfy.

## What the package contains

| Module | Contents |
| --- | --- |
| `ggufhdr` | The header: the magic, the version, the byte-order test, the two count widths, and the alignment rule. |
| `ggufval` | The thirteen metadata value types, a value as a novo-lang value, and one key/value entry. |
| `ggufmeta` | The metadata table: lookup by key at a type, the named keys every model declares, and the architecture-scoped key names. |
| `ggufinfo` | One tensor's entry: its rank, its element and row counts, its size in bytes, and the range its data occupies. |
| `ggufquant` | Every GGML tensor type, with the weights per block and the bytes per block that size a tensor. |
| `ggufrange` | A byte offset and a byte count, and the arithmetic over them: slicing, joining, containment and overlap. |
| `ggufread` | Reading an index: the asking reader, the whole-image form, the sequential form, and the index they all answer. |
| `ggufwrite` | Writing an index back: entries in, tensor entries in, the planned offsets out, and the bytes of the index. |
| `gguferr` | What a file could not be read as, with the offset, count, code or key that says why. |

## How to choose an entry point

There are three ways to get an index, and they differ in who reads the
bytes.

**`ggufread.parse_index` takes the whole file.** Use it for a small model,
a memory-mapped file, or a test.

**`ggufread.read_index` takes a source that can go forwards.** The header
and both tables are contiguous from byte 0, so a sequential source is
enough. It costs whatever the source costs: nothing for a buffer, `[io]`
for a file, `[io, net]` for a socket.

**`ggufread.reader`, `need`, `offer` and `index_of` ask for bytes.** The
reader answers a range, the caller reads exactly those bytes and hands them
back, and the loop repeats until `is_done`. Use it when the source is an
HTTP range request, an accelerator's upload queue, or anything else that is
not a stream. Nothing in the format records where the metadata table ends,
so neither "the whole file" nor "the first N bytes" is a promise a caller
could keep in advance.

For a tensor's bytes there are two ways in. **`ggufread.tensor_range` takes
the index and a name** and answers the range, which is what a loader wants.
**`ggufinfo.tensor_range` takes one entry and the blob's start**, for a
caller holding entries it filtered or sorted itself.
`ggufinfo.row_range` and `ggufquant.block_range` are the same question at
a finer grain.

For a metadata value there are three. `ggufmeta.int_at` and its four
siblings answer a `Result` that names the key and both type names when the
type is wrong. `ggufmeta.int_or` and its siblings answer a fallback
instead. `ggufmeta.find` answers the raw `GgufValue` for a caller that will
match on it.

## The rules a user needs

1. **The magic does not say which byte order the file is in.** The four
   bytes are `GGUF` in file order in both. The version field decides: a
   version whose low sixteen bits are zero was read backwards, because no
   real version is a multiple of 65536. `ggufhdr.detect_order` is that
   test, and it is the reference implementation's own.
2. **Version 1 writes 32-bit counts and versions 2 and 3 write 64-bit
   ones.** The same widening applies to every string length and every array
   length in the tables. `ggufhdr.count_width` is the one place that fact
   lives.
3. **A tensor entry's offset is relative to the data blob and a
   `GgufRange` is absolute.** `GgufTensorInfo.offset` keeps the number the
   file stored, so a round trip through `ggufwrite` writes back what was
   read. Every function that answers a position adds the blob's start. A
   reader that mixes the two gets its first tensor right and its last one
   wrong by megabytes.
4. **The data blob starts at the first multiple of the alignment after the
   tensor-info table.** The alignment is `general.alignment` or 32.
   `GgufIndex.alignment` and `GgufIndex.data_at` are both resolved when the
   index is parsed, because every tensor range depends on them.
5. **A tensor's size is blocks, not bits per weight.**
   `ggufquant.block_elements` and `ggufquant.block_bytes` are the two
   numbers, and `row_bytes`, `tensor_bytes` and `block_range` are built on
   them. `bits_per_weight` is for reporting and is not the way to size
   anything.
6. **A row's length must be a multiple of the block's weight count.** A
   Q4_K tensor whose first dimension is 100 does not exist, because 100 is
   not a multiple of 256.
7. **The architecture prefixes its own keys.** A Llama model's layer count
   is `llama.block_count` and a Qwen2 model's is `qwen2.block_count`, so a
   reader reads `general.architecture` first and builds every other key
   name from it. `ggufmeta.arch_key` is that concatenation, and the eight
   named helpers over it cover the hyperparameters every decoder-only
   architecture declares.
8. **`general.file_type` is a hint and not a fact.** It names what the
   quantiser was asked for. The tensor-info table names what each tensor
   actually is, and a mixed quantisation makes the two disagree by design.
   `ggufinfo.type_histogram` is what shows a user that a `Q4_K_M` file is
   four different quantisations, which it is.
9. **An unknown metadata value-type code stops the read.** The format has
   thirteen, and a value whose length this reader cannot compute is a value
   it cannot step over. `GgufBadValueType` carries the code.
10. **Five tensor type codes were withdrawn from the format.** A file that
    uses one cannot be read. `ggufquant.withdrawn_codes` lists them and
    `withdrawn_name` gives the name, so the message says which type rather
    than reporting an unknown number.
11. **`GgufQ8_K` and `GgufQ8_1` are runtime types and are not written to
    files.** `ggufquant.is_file_type` is the predicate. A tensor declaring
    one is a file something built wrong.
12. **A duplicated metadata key answers its first entry.**
    `ggufmeta.duplicates` lists every key that appears more than once, so a
    tool can report the file rather than being refused it.
13. **An array may hold arrays, and the format bounds the nesting at
    nothing.** This package bounds it at `ggufval.max_array_depth`, which
    is eight, and answers `GgufArrayTooDeep` past it. Without a bound a
    file could ask a reader for a stack overflow.
14. **A 64-bit unsigned value with its top bit set does not fit.**
    novo-lang's `Int` is signed. `GgufTypeUint64` is the one metadata type
    whose full range does not fit, and such a value is
    `GgufUnsignedOverflow` rather than a negative number.
15. **`offer` takes exactly the bytes `need` asked for.** A short chunk is
    `GgufTruncated`, because a host that read fewer bytes than it asked for
    has hit the end of the file. A longer chunk is refused too, because the
    extra bytes have no position the reader can trust.
16. **`ggufinfo.check_layout` is a separate call from parsing.** A damaged
    checkpoint can be read and then be told it is damaged, which is what a
    repair tool needs.
17. **A misaligned tensor offset is refused.** llama.cpp reads one and
    addresses it wrongly. `ggufinfo.validate` is the check.

## What is not included

- **Dequantisation.** This package gives the block layout as numbers. It
  never turns a block back into weights, and it holds no tensor. An
  inference engine owns its kernels, and it needs nothing from here but the
  index and the two block numbers.
- **The weights, in the writer.** `ggufwrite` emits the index only. A tool
  editing a checkpoint's metadata copies the data blob through unchanged.
- **A tensor type.** `GgufTensorInfo` carries a name, dimensions, a type
  code and an offset. What the numbers mean is the caller's.
- **safetensors.** `std.safetensors` in the standard library reads that
  format.
- **A microcontroller build.** No module here is declared to build for a
  device with no heap allocator. A GGUF is addressed by 64-bit offsets into
  a file measured in gigabytes.

## Related packages

- [embeddings-nv](https://novo-lang.org/packages/embeddings-nv) is the
  arithmetic on what a model produced. The two packages sit at opposite
  ends of an inference run and never meet.
- [tokenizers-nv](https://novo-lang.org/packages/tokenizers-nv) builds a
  tokenizer from a vocabulary. A GGUF's `tokenizer.ggml.*` keys are where
  that vocabulary comes from, and `ggufmeta.key_tokens`,
  `key_merges` and `key_token_type` name them.
- [prompt-nv](https://novo-lang.org/packages/prompt-nv) renders a chat
  template. `ggufmeta.chat_template` reads the one the file carries.
- [ollama-nv](https://novo-lang.org/packages/ollama-nv) talks to a local
  Ollama server, which serves GGUF files. It never reads one itself.
- [elf-nv](https://novo-lang.org/packages/elf-nv) answers about a firmware
  image with the same offset-and-length shape, so a reader who has met one
  meets no surprise here. The two packages share the idea and no code.
- `std.safetensors` in the standard library loads `.safetensors` weights
  into tensors. It reads the other common checkpoint format, and it
  produces tensors rather than byte ranges.
- `std.hf` in the standard library fetches repository files from the
  HuggingFace Hub into a revision-pinned cache, which is where a GGUF
  usually comes from.
- `std.llm` in the standard library runs inference. It is the consumer a
  loader built on this package would feed.

## Tests

```bash
novo test tests/ggufquant_tests.nv     #  8 tests: the type codes and the block sizes
novo test tests/ggufindex_tests.nv     # 15 tests: the header, the tables and the ranges
novo test tests/ggufsurface_tests.nv   #  7 tests: the rest of the public surface
```

The block sizes come from `ggml-common.h`, one struct per type, and each
assertion writes out the sum rather than the total: `block_q4_K` is
`ggml_half d` plus `ggml_half dmin` plus `uint8_t scales[12]` plus
`uint8_t qs[128]`, which is 2 + 2 + 12 + 128. A wrong number is then
visible as a wrong addition. The header, the value type codes, the string
encoding, the tensor entry's field order, the maximum rank and the default
alignment come from the GGUF specification.

`gguf-py`'s `GGUFReader` is the oracle, and it arrives with the bodies: a
set of small published checkpoints read with it and with this package,
every key, every tensor offset and every size compared.

The tests compile today and fail at run, each on the
`not implemented: gguf-nv.<module>.<fn>` panic that is its body. That is
the expected state of an interface release. They turn green one at a time
as bodies land.

## Implementation status

| Item | Implemented |
| --- | --- |
| `ggufhdr.magic_bytes`, `.magic_le`, `.magic_range`, `.version_range`, `.probe_range` | no |
| `ggufhdr.parse_header`, `.is_gguf`, `.detect_order`, `.order_name` | no |
| `ggufhdr.supported_versions`, `.latest_version`, `.count_width`, `.header_bytes`, `.header_range` | no |
| `ggufhdr.default_alignment`, `.alignment_is_valid`, `.align_up` | no |
| `ggufval.value_type_code`, `.value_type_of_code`, `.value_type_name`, `.scalar_width` | no |
| `ggufval.is_fixed_width`, `.is_integer_type`, `.is_unsigned_type`, `.max_array_depth` | no |
| `ggufval.value_kind`, `.value_depth`, `.encoded_bytes`, `.kv_bytes`, `.describe` | no |
| `ggufval.as_int`, `.as_float`, `.as_bool`, `.as_str`, `.as_list`, `.as_str_list`, `.as_int_list`, `.as_float_list` | no |
| `ggufmeta.of_entries`, `.entries_of`, `.count`, `.keys`, `.duplicates`, `.find`, `.has`, `.prefixed` | no |
| `ggufmeta.int_at`, `.float_at`, `.bool_at`, `.str_at`, `.list_at`, `.str_list_at`, `.int_list_at`, `.float_list_at` | no |
| `ggufmeta.int_or`, `.float_or`, `.bool_or`, `.str_or` | no |
| `ggufmeta`'s thirteen named key constants, and `arch_key` with its eight scoped keys | no |
| `ggufmeta.architecture`, `.alignment`, `.chat_template` | no |
| `ggufinfo.max_rank`, `.rank`, `.element_count`, `.row_elements`, `.row_count` | no |
| `ggufinfo.tensor_bytes`, `.tensor_range`, `.row_range`, `.entry_bytes`, `.validate` | no |
| `ggufinfo.find`, `.names`, `.by_offset`, `.with_prefix`, `.data_bytes`, `.check_layout`, `.type_histogram` | no |
| `ggufquant.type_code`, `.type_of_code`, `.type_name`, `.type_of_name`, `.known_types` | no |
| `ggufquant.block_elements`, `.block_bytes`, `.bits_per_weight` | no |
| `ggufquant.is_quantized`, `.is_k_quant`, `.is_iq_type`, `.is_ternary`, `.is_file_type` | no |
| `ggufquant.withdrawn_codes`, `.withdrawn_name` | no |
| `ggufquant.row_bytes`, `.tensor_bytes`, `.blocks_in_row`, `.block_range` | no |
| `ggufrange.range`, `.empty`, `.range_end`, `.range_fits`, `.slice`, `.sub`, `.join`, `.overlaps` | no |
| `ggufread.reader`, `.need`, `.offer`, `.is_done`, `.index_of` | no |
| `ggufread.stage_name`, `.entries_seen`, `.tensors_seen` | no |
| `ggufread.parse_index`, `.read_index`, `.index_range`, `.tensor_range`, `.data_range`, `.check_index` | no |
| `ggufwrite.writer`, `.default_writer`, `.rewriter` | no |
| `ggufwrite.push_kv`, `.push_kvs`, `.set_kv`, `.drop_kv`, `.push_tensor`, `.push_tensors` | no |
| `ggufwrite.index_bound`, `.data_at`, `.planned_offsets`, `.finish`, `.write_into` | no |
| `gguferr.is_corrupt`, `.fault_offset`, `.fault_stage`, `GgufFault.message` | no |

## Licence

Apache-2.0. See `LICENSE`.

<!-- docs/writing-a-readme.md is the style guide for this page. -->
