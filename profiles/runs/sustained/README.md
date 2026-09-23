# Sustained release-gate run (#38)

- Workload: `bench_mtp.py --prompt "Explain photosynthesis..." --max-tokens 50 --skip-spec`
- Monitor: `/tmp/sustained_monitor.sh` → samples.csv (CPU%, mem MB, free%, every 30s)
- Status: run exceeded expected wall (expert I/O thrashing at cap 64); terminated after >90 min with partial decode progress; monitor samples retained for memory/CPU stability.
- Thermal: `pmset -g therm` reported no thermal/performance warning recorded; `powermetrics` needs sudo (not run).
- Golden quality: golden-pass.json (PASS), golden-fail.json (FAIL on perturb).
