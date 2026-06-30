# Post-freeze environment record

Status: POST_FREEZE_ENVIRONMENT_RECORD_NOT_CONFIRMATORY

Recorded: 2026-06-30T08:36:27Z

## python --version

```text
zsh:6: command not found: python
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
      Model Number: REDACTED
      Chip: Apple M2 Max
      Total Number of Cores: 12 (8 performance and 4 efficiency)
      Memory: 96 GB
      System Firmware Version: 13822.81.10
      OS Loader Version: 11881.140.96
      Serial Number (system): REDACTED
      Hardware UUID: REDACTED
      Provisioning UDID: REDACTED
      Activation Lock Status: REDACTED

Software:

    System Software Overview:

      System Version: macOS 15.7.4 (24G517)
      Kernel Version: Darwin 24.6.0
      Boot Volume: Macintosh HD
      Boot Mode: Normal
      Computer Name: Larry’s MacBook Pro (6)
      User Name: Larry (larry)
      Secure Virtual Memory: Enabled
      System Integrity Protection: Enabled
      Time since boot: 38 days, 8 hours, 55 minutes

```

## git status --short

```text
?? reports/post_freeze_environment_record.md
```

## git rev-parse HEAD

```text
69028ae3b8c7829067bdbca03042b141bca4f8e4
```
