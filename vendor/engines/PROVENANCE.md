# Media engine and trainer source

Derived from a private fork at c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3, cosmos_windows. Only Python engine/trainer modules are retained. Documentation, backups, tests, data, weights and private history are excluded. Machine-specific fallback paths were cleared. MIT for owner-authored code; third-party licenses must accompany reused third-party modules.

Local Anima fixes load explicit adapters through the pipeline's native-name converter for both the transformer and text conditioner, materialize adapter tensors alongside a streamed meta-device base, and bind/evict only the wrapped base weights. Adapter shape and finite-value checks fail visibly before sampling. The shared adapter helper accepts an optional low-memory loader argument; other callers retain their existing behavior.
