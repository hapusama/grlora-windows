# Source demod provenance

These CSV files were regenerated from the three high-SNR USRP captures before
the added-noise experiment. For each dataset `<D>`, the commands were:

```powershell
python -B scripts\run_header_first_demod.py `
  -i ..\data\USRP_IQ\<D>.bin `
  -s data\weak_sync_chain\sync_chain\<D>_sync_chain.csv `
  -o data\experiments\real_iq_payload_noise_cfr_nus_20260720\source_demod\<D>_header_first_symbols.csv `
  --frames-output data\experiments\real_iq_payload_noise_cfr_nus_20260720\source_demod\<D>_header_first_frames.csv `
  --cfo-correction-mode continuous

python -B scripts\verify_payload_codec_alignment.py `
  --gt-symbol-csv data\experiments\real_iq_payload_noise_cfr_nus_20260720\source_demod\<D>_header_first_symbols.csv `
  --sf 10 `
  --output-csv data\experiments\real_iq_payload_noise_cfr_nus_20260720\source_demod\<D>_codec_audit.csv
```

`<D>` was each of `0_0_0_10_14_8`, `0_0_0_10_14_16`, and
`0_0_0_10_14_32`. The audit found 28/28 CRC-valid packets, zero mismatches in
840 deterministic payload-prefix symbols, and 140 padding-suffix symbols that
were reported but excluded from SER.
