# Streaming llama

Derived from a private fork at `c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3`. Upstream llama.cpp (MIT). Exact upstream base identification is pending. This export retains core model/ggml sources, the server UI, LoRA exporter, original V4.1 producer and native session driver/codec. It excludes private history, evidence, weights and unrelated tools. Changes: portable diagnostic destinations and narrowed build targets. See the retained LICENSE and AUTHORS.

Portable diagnostic source changes are published at `79d2d481e59875b4014552454ff876614c4afb50`. The clean export also restores required include/link dependencies, narrows build targets, and provides a thin weight-fetcher DLL with correct static Windows declarations. These build changes do not replace the streaming arithmetic.
