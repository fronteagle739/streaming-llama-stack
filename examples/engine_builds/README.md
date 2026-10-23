# Stack source builds

The Mac completed a fresh GitHub clone and native build with no source edits. Windows completed the native CUDA, four streaming targets, producer profile preflight and chat UI builds after a committed source-closure correction exposed by the initial clone. A subsequent literal fresh clone of the published installer fix passed the single `slstack install --all` command and its readiness check, with no manual fixes or prior compiler outputs restored. See RESULTS.json for source identities, binaries and physical-memory receipts.

The first Windows compile stopped before CUDA/model loading because three provider include dependencies were missing from the export. Those exact headers were added from the same frozen provider. The Mac engine completed successfully; its enclosing controller then blocked on a large stdout receipt. The completed receipt was recovered after independently confirming that no owned Mac guardian remained. These failures remain distinct from the successful build receipts.

Existing accepted model parity measurements are linked in ../engine_parity/README.md and are reused. No completed model test was repeated for this build gate.
