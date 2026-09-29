# Virtual-phase SFD synchronization: paired OTA+AWGN audit

Only the second-SFD integer-CFO estimator changes between conditions. The virtual condition coherently combines OSR polyphase views using Savaux's LoRa chirp-wrap phase law. Payload truth is used only for post-hoc evaluation, never to choose a synchronization result.

| Es/N0 (dB) | chip integer-CFO correct | virtual integer-CFO correct | CFO rescues / regressions | chip / virtual local | chip / virtual paired SER | payload rescues / regressions |
|---:|---:|---:|---:|---:|---:|---:|
| 13 | 0.404 | 0.511 | 14 / 9 | 0.660 / 0.979 | 0.3637 / 0.1144 | 376 / 1 |
