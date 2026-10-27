# Windows ds4 provenance

Derived from [antirez/ds4 at `a0510c2ac2bec2cbffe730fa728fb7d3880f93ec`](https://github.com/antirez/ds4/commit/a0510c2ac2bec2cbffe730fa728fb7d3880f93ec), the merge-base between the exported Windows source ancestry and upstream ds4, MIT.

The Windows port also incorporates [mishal-harbi/ds4-win-NovaStar at `8e3cfa00e60c9ba45acaf0c09e47054ca236521c`](https://github.com/mishal-harbi/ds4-win-NovaStar/commit/8e3cfa00e60c9ba45acaf0c09e47054ca236521c), the merge-base of the original NovaStar-derived checkout and its public branch. Subsequent selective ports include V4.1 contributions from [IngeniousIdiocy/ds4-v41-m3ultra](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra); the retained Windows graph does not establish a direct V4.1 branch point. The historical `IngeniousIdiocy/stack-backend` repository could not be resolved to a verifiable upstream base.

Changes: DirectStorage expert streaming/cache policies, native KV handoff, four-phase quality controls, dense-output adapters/training, OpenAI timing fields, standalone build closure and memory guards. Original licences, embedded-library attribution and third-party notices are retained.
