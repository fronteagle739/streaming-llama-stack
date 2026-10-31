| config | 0 KB p50 / p99 (us) | 8 KB p50 / p99 (us) | 14 KB p50 / p99 (us) | 28 KB p50 / p99 (us) | runs |
|---|---:|---:|---:|---:|---:|
| tcp-plain-fwd | - | 319 / 376 | 323 / 436 | 452 / 541 | 2 |
| tcp-nodelay-fwd | - | 320 / 431 | 385 / 477 | 518 / 640 | 2 |
| tcp-nodelay-qa-fwd | - | 319 / 384 | 321 / 411 | 448 / 519 | 2 |
| tcp-tuned-fwd | - | 194 / 298 | 207 / 336 | 322 / 421 | 2 |
| tcp-tuned-4mbuf-fwd | - | 194 / 282 | 213 / 365 | 325 / 438 | 2 |
| udp-block-fwd | - | 322 / 479 | 455 / 630 | 762 / 977 | 2 |
| udp-spin-fwd | - | 184 / 301 | 299 / 365 | 351 / 490 (retx 509) | 2 |
| udp-spin-ipfrag-fwd | - | 186 / 309 (retx 2) | 192 / 328 | 313 / 535 (retx 2569) | 2 |
| tcp-plain-rev | - | 314 / 437 | 316 / 461 | 444 / 524 | 2 |
| tcp-nodelay-rev | - | 314 / 411 | 314 / 709 | 412 / 2276 | 2 |
| tcp-nodelay-qa-rev | - | 314 / 2232 | 313 / 2423 | 382 / 2538 | 2 |
| tcp-tuned-rev | - | 183 / 1898 | 295 / 1690 | 314 / 1786 | 2 |
| tcp-tuned-4mbuf-rev | - | 183 / 1610 | 252 / 1667 | 316 / 1374 | 2 |
| udp-block-rev | - | 312 / 474 (retx 393) | 504 / 631 (retx 1) | 716 / 897 (retx 10) | 2 |
| udp-spin-rev | - | 308 / 336 (retx 7) | 318 / 363 (retx 19) | 432 / 476 (retx 11) | 2 |
| udp-spin-ipfrag-rev | - | 182 / 323 (retx 1) | 312 / 339 (retx 8) | 314 / 427 (retx 33) | 2 |
| relay-nodelay-fwd | - | 395 / 488 | 448 / 556 | 572 / 703 | 2 |
| relay-tuned-fwd | - | 200 / 315 | 258 / 403 | 316 / 441 | 2 |
| relay-nodelay-rev | - | 376 / 464 | 441 / 562 | 516 / 662 | 2 |
| relay-tuned-rev | - | 182 / 356 | 313 / 380 | 322 / 520 | 2 |
| mac-loopback-nodelay | - | 61 / 123 | 62 / 127 | 77 / 108 | 1 |
| mac-loopback-relay-nodelay | - | 106 / 164 | 104 / 143 | 123 / 197 | 1 |
| mac-loopback-nodelayspinprio | - | 64 / 79 | 27 / 102 | 88 / 130 | 1 |
| mac-loopback-relay-nodelayspinprio | - | 110 / 244 | 33 / 222 | 97 / 116 | 1 |
| win-loopback-tuned | 14 / 30 | 17 / 37 | 18 / 36 | 21 / 40 | 1 |
| win-loopback-plain | 24 / 86 | 27 / 98 | 29 / 107 | 32 / 108 | 1 |
