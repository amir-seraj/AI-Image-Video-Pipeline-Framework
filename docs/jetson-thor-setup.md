# Jetson AGX Thor Setup for ML Inference

Hardware and system configuration for running large diffusion models (Wan 2.1 14B, etc.) on the NVIDIA Jetson AGX Thor Developer Kit.

## Hardware Specs

| Component | Specification |
|-----------|--------------|
| SoC | NVIDIA Thor (Blackwell, SM 110) |
| CPU | 14-core Arm (Cortex-A520) @ 2.6 GHz |
| GPU | Blackwell GPU, 1386/1557 MHz |
| Memory | 128 GB unified (CPU+GPU shared) |
| CUDA | 13.0, Driver 580.00 |
| EMC | DDR5x @ 4266 MHz |
| Power Mode | 120W (NV Power Mode 1) |

## Power & Thermal Management

### The Problem

The default NVIDIA fan profile (`nvfancontrol`) is tuned for quiet desktop use, not sustained ML inference. It uses a TMARGIN-based fan curve where:

- **TMARGIN** = `max_temp (115C) - current_temp`
- At 68C (TMARGIN=47), the default profile sets the fan to just **30% (PWM 77)**
- The fan only ramps up when temps exceed ~90C

Under heavy GPU load (model loading, inference), this causes:
1. Temps rise quickly due to inadequate cooling
2. The system draws more current to maintain clock speeds
3. Current exceeds the 120W power budget rail limits
4. BPMP firmware triggers **overcurrent throttling**, clocking down the GPU/CPU
5. Inference becomes slow or the process gets OOM-killed

### The Fix

#### 1. Aggressive Fan Profile (persistent across reboots)

We modified the board-specific `nvfancontrol` config to keep the fan spinning faster at all temperatures:

**Config file:** `/etc/nvpower/nvfancontrol/nvfancontrol_p3834_0008_p4071_0000.conf`

Original backup: `/etc/nvpower/nvfancontrol/nvfancontrol_p3834_0008_p4071_0000.conf.bak`

```
# TMARGIN   HYST   PWM   RPM
# Original (quiet desktop):
#   45+      0      77    1750   (30% at idle)
#   35       0     102    2300
#   29       0     140    2900
#   24       0     192    4170
#   15       0     255    5371   (100% only near thermal limit)

# ML inference profile (current):
  0          0     255    5371   (100% at thermal limit)
  15         0     255    5371
  24         0     255    5371
  29         0     230    4800
  35         0     200    4170
  45         0     180    3800   (71% at idle — was 30%)
  60         0     160    3400
  115        0     140    2900   (55% minimum — was 30%)
```

To apply changes: `sudo systemctl restart nvfancontrol`

To revert: `sudo cp /etc/nvpower/nvfancontrol/nvfancontrol_p3834_0008_p4071_0000.conf.bak /etc/nvpower/nvfancontrol/nvfancontrol_p3834_0008_p4071_0000.conf && sudo systemctl restart nvfancontrol`

#### 2. Max Clocks on Boot (persistent via systemd)

A systemd service runs `jetson_clocks` on boot, which:
- Locks all 14 CPU cores to max frequency (2.6 GHz)
- Locks GPU to max frequency (1386/1557 MHz)
- Locks EMC (memory bus) to max (4266 MHz)
- Disables CPU idle states for lower latency

**Service file:** `/etc/systemd/system/jetson-clocks.service`

```ini
[Unit]
Description=Maximize Jetson clocks for ML inference
After=nvpmodel.service nvfancontrol.service

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

To check status: `sudo jetson_clocks --show`

To disable (save power when not doing inference):
```bash
sudo systemctl disable jetson-clocks.service
sudo systemctl stop jetson-clocks.service
```

### Results

| Metric | Before | After |
|--------|--------|-------|
| Fan speed at idle | 30% (PWM 77) | 63% (PWM 160) |
| GPU temp at idle | 68C | 37C |
| CPU temp at idle | 63C | 36C |
| Overcurrent throttling | Frequent under load | None observed |

## Memory Considerations

The 128 GB unified memory is shared between CPU and GPU. Key observations running Wan 2.1 I2V-14B:

| Configuration | GPU Memory After Load | Notes |
|--------------|----------------------|-------|
| BF16 (default) | ~44 GB | Works for small frame counts |
| FP8 (torchao) | ~31 GB | 30% reduction via `Float8WeightOnlyConfig` |

**Important:** Even with 128 GB total, generating 81 frames at 720p with 50 inference steps requires significant activation memory on top of model weights. The process was OOM-killed at ~133 GB virtual memory.

Recommendations:
- Close other GPU-consuming processes before running large inference jobs
- Start with fewer frames (17) to verify the pipeline works, then scale up
- Monitor memory with `watch -n1 'cat /proc/meminfo | grep -E "MemTotal|MemFree|MemAvailable"'`
- Use `torch.cuda.max_memory_allocated()` in scripts to track peak GPU usage

## Useful Commands

```bash
# Check temps
for z in /sys/devices/virtual/thermal/thermal_zone*; do
    echo "$(cat $z/type): $(($(cat $z/temp) / 1000))C"
done

# Check fan speed (0-255)
cat /sys/devices/platform/pwm-fan/hwmon/hwmon1/pwm1

# Check power draw per rail
for label_file in /sys/bus/i2c/drivers/ina3221/2-0040/hwmon/hwmon6/*label; do
    base=$(echo "$label_file" | sed 's/label//')
    name=$(cat "$label_file")
    voltage=$(cat "${base}input" 2>/dev/null)
    current_file=$(echo "$label_file" | sed 's/in\([0-9]*\)_label/curr\1_input/')
    current=$(cat "$current_file" 2>/dev/null)
    power=$(( voltage * current / 1000000 ))
    echo "$name: ${voltage}mV x ${current}mA = ${power}W"
done

# Check current power mode
sudo nvpmodel -q

# Check all clocks
sudo jetson_clocks --show

# Kernel throttle/OOM messages
sudo dmesg | grep -iE "throttl|oom|over.?current" | tail -20

# Force fan to max immediately (bypasses nvfancontrol)
sudo bash -c 'echo 255 > /sys/devices/platform/pwm-fan/hwmon/hwmon1/pwm1'
```

## Network: 10GbE Link Flapping

The Thor dev kit has 4x 10GbE ports (`mgbe0_0` to `mgbe3_0`). When no cable is connected, the firmware continuously retries the link (~1/sec), flooding `dmesg` with:
```
nvethernet a808e10000.ethernet: [xpcs_lane_bring_up][827] PCS block lock SUCCESS
```

This causes instability — under high power load the link retries can glitch the 1GbE interface (`enP2p1s0`) and drop SSH connections.

**Fix:** A systemd service disables unused 10GbE interfaces on boot:

**Service file:** `/etc/systemd/system/disable-mgbe.service`

To re-enable a 10GbE port (if you plug in a cable): `sudo ip link set mgbe0_0 up`

## Clocks: jetson_clocks

**Do NOT run `jetson_clocks` persistently.** It locks all CPU/GPU/EMC to max with idle states disabled, which increases idle power draw enough to cause overcurrent and ethernet instability. Only run it manually before inference if needed:

```bash
# Before a big inference run (optional)
sudo jetson_clocks

# Check current state
sudo jetson_clocks --show
```

The aggressive fan profile alone is sufficient for stable operation.

## File Locations

| File | Purpose |
|------|---------|
| `/etc/nvpower/nvfancontrol/nvfancontrol_p3834_0008_p4071_0000.conf` | Board-specific fan profile (active) |
| `/etc/nvpower/nvfancontrol/nvfancontrol_p3834_0008_p4071_0000.conf.bak` | Original fan profile backup |
| `/etc/nvfancontrol.conf` | Generic fan config (not used by service) |
| `/etc/systemd/system/disable-mgbe.service` | Disables unused 10GbE ports on boot |
| `/etc/systemd/system/nvfancontrol.service` | Fan control daemon |
