# Third-party notices

The repository is MIT (see [LICENSE](LICENSE)). The components below carry
their own terms and are **not** covered by it. This file exists so a
redistributor can tell, without reading the whole tree, what obligations come
with a build.

## rax.c / rax.h — BSD 3-Clause

```
Copyright (c) 2017-2026, Salvatore Sanfilippo <antirez at gmail dot com>
All rights reserved.
```

Redistribution in source and binary form is permitted with the copyright
notice, the conditions list, and the disclaimer retained; the third clause
forbids using the Redis name or contributor names to endorse derived products
without written permission. The full text is in the header of `rax.c` and must
stay there.

**Obligation when shipping a binary:** reproduce that notice in the
documentation or other materials distributed with it. `ds4-server`, `ds4`,
`ds4-agent` and `ds4-bench` all link `rax.c`, so this applies to every binary
this tree produces.

## ggml / llama.cpp derived CUDA kernels — MIT

`cuda/mmq/` is vendored from the ggml/llama.cpp CUDA backend: `ds4_mmq.cu`,
`ds4_mmq_d2r.cu`, `ds4_repack.cu`, `ds4_ggml_stubs.cu`, `mmid.cu`, `mmvq.cu`,
`quantize.cu` and their headers, plus `mma.cuh`, `common.cuh`, `ggml-common.h`,
`ggml-impl.h`, `ggml.h`, `ggml-cuda.h`.

These are MIT and covered by the `Copyright (c) 2023-2026 The ggml authors`
line already in [LICENSE](LICENSE). No separate obligation beyond retaining
that notice.

## Microsoft DirectStorage SDK — Microsoft licence, NOT MIT

`external/directstorage/` contains Microsoft redistributables and headers:

```
external/directstorage/native/bin/x64/dstorage.dll
external/directstorage/native/bin/x64/dstoragecore.dll
external/directstorage/native/include/dstorage.h
external/directstorage/native/include/dstorageerr.h
external/directstorage/native/lib/x64/dstorage.lib
```

`windows_build.ps1` copies both DLLs next to every executable it builds, so
**they ship with the binaries**. They are Microsoft's, licensed under
Microsoft's terms, and are not placed under the MIT licence by this repository.

The licence text is vendored beside them:

```
external/directstorage/LICENSE.txt        Microsoft Software License Terms - DirectStorage SDK
external/directstorage/LICENSE-CODE.txt   MIT, for the sample/source portions of the package
external/directstorage/NOTICES.txt        third-party notices for components inside the SDK
```

**Version, established by hash rather than assumed.** These came from
`Microsoft.Direct3D.DirectStorage` **1.3.0**, and both vendored DLLs are
SHA-256 identical to that package's:

```
dstorage.dll      CAE65787F61514BEC6F012F44546E68B19E39D8854A9AB52AF813370D44B272A
dstoragecore.dll  CF85B19A92A8276F8BACA0DB9BBCD60735BAD50768DEB303D310E84D5F4FDD11
```

That matters because the terms are version-specific: `dstorage.dll.122.bak` and
`dstoragecore.dll.122.bak` in the same directory are the superseded 1.2.2
binaries, and a licence copied for the wrong version would be worse than none.
**If the DLLs are ever updated, re-fetch these three files with them** and
re-check the hashes.

### What redistribution actually obliges (LICENSE.txt §2)

Distributing the DLLs — which any binary built here does — requires that you:

* **add significant primary functionality** to them in your application (§2.b.i);
* **bind your distributors and end users** to terms protecting Microsoft at
  least as much as the SDK agreement does (§2.b.ii);
* **indemnify Microsoft** for claims arising from your application's
  distribution or use, except claims resting solely on the unmodified
  distributable code (§2.b.iii).

And you may not use Microsoft trademarks or trade dress in a way suggesting the
application is endorsed by Microsoft (§2.c.i), nor place the distributable code
under a licence that would require it to be disclosed in source form or made
modifiable (§2.c.ii). Note the last one is a genuine constraint on this project:
it rules out ever moving to a copyleft licence while these binaries ship in the
tree.

## linenoise — BSD 2-Clause

`linenoise_win32.c` is this fork's Windows replacement for the POSIX
`linenoise.c`. Where it derives from the original linenoise by Salvatore
Sanfilippo and Pieter Noordhuis, the BSD 2-Clause terms of that project apply
and its notice is retained in the file.

## Model weights are not covered by anything here

The GGUF files this engine runs are **not** part of this repository and carry
their own licences from their publishers. Nothing in [LICENSE](LICENSE) grants
any right to a model. Check the model card of whatever you load.
