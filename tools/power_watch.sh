#!/usr/bin/env bash
# How hard is the board being pushed? Counts the kernel's over-current alarms
# ("soctherm: OC ALARM") for a while and shows GPU clock, temperature and CPU load.
#
#     bash tools/power_watch.sh 120        # watch for 120 seconds (default 60)
#
# The alarms show that the board draws more current than its limit allows and slows
# itself down. A few are normal under load; many per second while the GPU works, or a
# board that reboots by itself ("TEGRA_POWER_ON_RESET"), means the power supply or cable
# is too weak or the load (engine size x frames per second) is too high: lower max_fps.
# Compare two settings by running this under each.
SECONDS_TO_WATCH="${1:-60}"
count() { dmesg 2>/dev/null | grep -c "OC ALARM"; }
gpu() { cat /sys/devices/gpu.0/devfreq/*/cur_freq 2>/dev/null | head -1; }
temp() {
    for z in /sys/devices/virtual/thermal/thermal_zone*; do
        [ "$(cat "$z/type" 2>/dev/null)" = "GPU-therm" ] && cat "$z/temp" && return
    done
    echo 0
}
start=$(count)
echo "watching for ${SECONDS_TO_WATCH}s (alarms so far: $start)"
end=$((SECONDS + SECONDS_TO_WATCH))
busy=0; n=0
while [ $SECONDS -lt $end ]; do
    f=$(gpu); [ "${f:-0}" -gt 600000000 ] && busy=$((busy + 1))
    n=$((n + 1)); sleep 0.2
done
now=$(count)
delta=$((now - start))
[ "$delta" -lt 0 ] && delta=0   # the kernel log is a ring: old lines can drop out while we watch
echo "over-current alarms in ${SECONDS_TO_WATCH}s: $delta"
echo "GPU at full clock in $((100 * busy / n))% of the samples, GPU temperature $(( $(temp) / 1000 )) C"
uptime
