# Full Shear Gap Environment Record

Status: POST_FREEZE_ENVIRONMENT_RECORD_NOT_CONFIRMATORY

## date

```text
Tue Jun 30 20:19:08 JST 2026
```

## python --version

```text
Python 3.11.9
```

## uv run python environment

```text
torch 2.12.0
transformers 5.12.0
mps True
platform macOS-15.7.4-arm64-arm-64bit
```

## system_profiler SPHardwareDataType SPSoftwareDataType

```text
Hardware:

    Hardware Overview:

      Model Name: MacBook Pro
      Model Identifier: Mac14,6
      Model Number: [REDACTED]
      Chip: Apple M2 Max
      Total Number of Cores: 12 (8 performance and 4 efficiency)
      Memory: 96 GB
      System Firmware Version: 13822.81.10
      OS Loader Version: 11881.140.96
      Serial Number (system): [REDACTED]
      Hardware UUID: [REDACTED]
      Provisioning UDID: [REDACTED]
      Activation Lock Status: [REDACTED]

Software:

    System Software Overview:

      System Version: macOS 15.7.4 (24G517)
      Kernel Version: Darwin 24.6.0
      Boot Volume: Macintosh HD
      Boot Mode: Normal
      Computer Name: [REDACTED]
      User Name: [REDACTED]
      Secure Virtual Memory: Enabled
      System Integrity Protection: Enabled
      Time since boot: 38 days, 11 hours, 38 minutes
```

## git status --short --branch

```text
## main...origin/main
 M analysis/validation/common.py
?? analysis/diagnostics/full_transport_shear.py
?? analysis/diagnostics/shear_gap_regression.py
?? reports/full_shear_gap_environment_record.md
?? reports/full_transport_shear_diagnostic.checkpoint.jsonl
?? reports/full_transport_shear_diagnostic.json
?? reports/full_transport_shear_diagnostic.md
```

## git rev-parse HEAD

```text
01a68f077d63ff75d1758f0f68c5b22071a021cc
```
