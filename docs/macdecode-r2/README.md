# Mac decode evidence

These receipts retain measurements, options, failures, complete generated token
IDs and diagnostic logit hashes. Host-specific paths and the account name are
replaced by portable placeholders for publication. Full original receipts remain
in the recoverable task archive. No model weights or credentials are included.

`models.json` includes the original conversion SHA and a separate final drafter
metadata receipt. The final receipt supersedes intermediate drafter SHAs; its
tensor bytes are unchanged. `summary.json` separates warmup/proof requests from
timed requests. Serial rates exclude the first token; DSpark rates include the
complete decode. Candidate controls stay opt-in while release qualification is
incomplete.
