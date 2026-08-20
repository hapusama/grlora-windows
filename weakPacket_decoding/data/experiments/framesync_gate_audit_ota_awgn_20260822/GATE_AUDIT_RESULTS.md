# Decoder-aware FrameSync gate audit

- Packet trials: 80
- Strict-invalid but 16/16-symbol decodable: 15
- Those failing the all-preamble-bin0 gate: 15/15
- Decoder-aware policy: require a valid frame location and valid netID; defer the all-bin0 check to downstream FEC/CRC.

| Es/N0 | strict SER | decoder-aware SER | any-estimate SER | strict PDR | decoder-aware PDR | oracle-accept PDR | decoder-aware false rejects | decoder-aware nondecode accepts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0.8984 | 0.6094 | 0.4375 | 0.000 | 0.062 | 0.125 | 1 | 6 |
| 13 | 0.5742 | 0.3281 | 0.2070 | 0.312 | 0.500 | 0.562 | 1 | 3 |
| 14 | 0.4375 | 0.1875 | 0.1875 | 0.562 | 0.812 | 0.812 | 0 | 1 |
| 15 | 0.2500 | 0.0000 | 0.0000 | 0.750 | 1.000 | 1.000 | 0 | 0 |
| 16 | 0.0625 | 0.0000 | 0.0000 | 0.938 | 1.000 | 1.000 | 0 | 0 |

`accepted_nondecode_packets` is a compute/latency cost, not an undetected payload delivery when CRC remains mandatory.

## Failure counts

- strict_invalid_decodable / framesync_netid: 2
- strict_invalid_decodable / framesync_preamble_all_bin0: 15
- strict_invalid_nondecode / framesync_netid: 8
- strict_invalid_nondecode / framesync_preamble_all_bin0: 13
- strict_invalid_nondecode / locator_preamble_stability: 3
- strict_invalid_nondecode / locator_sfd: 6
- strict_invalid_nondecode / locator_sync1: 1
- strict_invalid_nondecode / locator_sync2: 1
