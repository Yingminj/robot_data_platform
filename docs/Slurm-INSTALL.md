# Slurm 26.05.2 DEB installation notes

**English** | [简体中文](Slurm-INSTALL.zh-CN.md)

This document only covers bringing every Ubuntu 22.04 `amd64` host onto the same cgroup v2-capable Slurm 26.05.2. Cluster node parameters, Munge distribution and startup order are covered in [Slurm cluster finalization](06-cluster-finalization.md).

**The whole cluster must use the same package version and the same build artifacts.** When versions are mixed, the symptom is node authentication or protocol incompatibility, and the error does not point at the version. When adding a node, use the same `artifacts/` directory; do not fetch packages from somewhere else on the fly.

Build artifacts in the repository:

```text
artifacts/slurm-26.05.2-jammy-amd64/
├── SHA256SUMS
├── debs/
├── metadata/
└── source/
```

The compressed archive and its checksum:

```text
artifacts/slurm-26.05.2-jammy-amd64.tar.gz
artifacts/slurm-26.05.2-jammy-amd64.tar.gz.sha256
```

## 1. Scope and installation order

The recommended order:

```text
run scripts/10-install-management.sh on mgmt01 first
run scripts/20-install-gpu-node.sh on every GPU worker first
→ install Slurm 26.05.2 from this page on every host
→ run slurmd -C on every host to collect hardware parameters
→ render and install the final Slurm configuration
```

Step 3 must be complete on **every** node before step 4 begins: rendering needs each machine's real hardware parameters, and one missing machine makes a complete `slurm.conf` impossible.

The reason is that the role scripts prepare accounts, directories, NFS and other system dependencies, but the Slurm shipped with Ubuntu 22.04 is old. The final version must be the one verified on this page.

After installing 26.05.2, avoid any operation that explicitly installs Ubuntu's `slurm-wlm`. If you really have to re-run `10`/`20`, first simulate the transaction with `apt -s` and confirm it will not remove `slurm-smd*`:

```bash
sudo apt -s install slurm-wlm
```

If the simulation would delete or replace `slurm-smd*`, do not confirm the transaction.

## 2. Build information

- upstream source: SchedMD Slurm 26.05.2;
- Debian package version: `26.05.2-1`;
- build target: Ubuntu 22.04 Jammy, `amd64`;
- includes `cgroup_v2.so`, `task_cgroup.so` and `proctrack_cgroup.so`;
- uses MUNGE authentication;
- includes the NVIDIA NVML GPU detection plugin.

NVML loads `libnvidia-ml.so.1` from the NVIDIA driver at runtime; these DEBs are not tied to a specific driver branch.

## 3. Put the artifacts on every host

Every host needs the complete `artifacts/slurm-26.05.2-jammy-amd64` directory. If a worker's copy of the repository does not have it, copy the archive from `mgmt01` (substituting that node's SSH target):

```bash
scp \
  artifacts/slurm-26.05.2-jammy-amd64.tar.gz \
  artifacts/slurm-26.05.2-jammy-amd64.tar.gz.sha256 \
  snorlax@192.168.100.215:~/
```

Verify the outer archive on that worker:

```bash
cd ~
sha256sum -c slurm-26.05.2-jammy-amd64.tar.gz.sha256
tar -xzf slurm-26.05.2-jammy-amd64.tar.gz
```

If every host already has the full repository, use the already-extracted directory in it; there is no need to extract again.

## 4. Verify the DEBs

Enter the artifact directory:

```bash
cd artifacts/slurm-26.05.2-jammy-amd64
sha256sum -c SHA256SUMS
```

If it was extracted into the home directory instead:

```bash
cd ~/slurm-26.05.2-jammy-amd64
sha256sum -c SHA256SUMS
```

Any checksum failure means stopping the installation and copying the files again.

## 5. Back up the old configuration

Run on every host:

```bash
sudo systemctl stop slurmctld slurmd 2>/dev/null || true

if sudo test -d /etc/slurm; then
  sudo cp -a \
    /etc/slurm \
    "/etc/slurm.backup.$(date +%Y%m%d-%H%M%S)"
fi
```

Do not delete `/etc/munge/munge.key`. The key generated on `mgmt01` by the role script is the single source for the cluster.

## 6. Install the four core packages

Run the same command in the artifact directory on every host:

```bash
sudo apt update
sudo apt install -y munge \
  ./debs/slurm-smd_26.05.2-1_amd64.deb \
  ./debs/slurm-smd-client_26.05.2-1_amd64.deb \
  ./debs/slurm-smd-slurmctld_26.05.2-1_amd64.deb \
  ./debs/slurm-smd-slurmd_26.05.2-1_amd64.deb
```

Use `apt install ./debs/...`; do not bypass dependency resolution with `dpkg -i`. If the transaction plans to remove critical system components, cancel it first and investigate the package conflict.

The current cluster does not need `slurmdbd`, `slurmrestd`, the development packages, PAM, NSS, PMI, Sview or Sackd.

## 7. Post-installation version check

Run on every host:

```bash
/usr/sbin/slurmctld -V
/usr/sbin/slurmd -V
dpkg-query -S /usr/sbin/slurmd
dpkg-query -S /usr/sbin/slurmctld
```

The first two are expected to be `26.05.2`, with the binaries owned by `slurm-smd-slurmd` and `slurm-smd-slurmctld`. Even if old package names are still visible in the dpkg database, the actual binary version and ownership checks are what count.

**Do not use `sbatch --version` to judge the installation at this point.** It fails on a machine that has no `/etc/slurm/slurm.conf` yet:

```text
sbatch: error: resolve_ctls_from_dns_srv: res_nsearch error: Unknown host
sbatch: error: fetch_config: DNS SRV lookup failed
sbatch: fatal: Could not establish a configuration source
```

Newer Slurm loads a configuration before printing the version, and when it finds none it falls back to configless DNS SRV discovery, which this cluster does not use. This is **not** an installation failure and disappears once the cluster configuration is installed.

## 8. cgroup v2 and NVML checks

On every host:

```bash
stat -fc %T /sys/fs/cgroup
nvidia-smi
ldconfig -p | grep -E 'libnvidia-ml\.so\.1'
find /usr/lib -type f \
  \( -name cgroup_v2.so -o -name gpu_nvml.so \) \
  -print
```

Expected:

- the cgroup filesystem prints `cgroup2fs`;
- `nvidia-smi` works;
- both `cgroup_v2.so` and `gpu_nvml.so` are found.

The cgroup configuration this repository ultimately installs is:

```ini
CgroupPlugin=autodetect
ConstrainCores=yes
ConstrainRAMSpace=yes
ConstrainDevices=yes
ConstrainSwapSpace=yes
```

The GPU configuration is a single card at `/dev/nvidia0`:

```ini
AutoDetect=nvml
Name=gpu File=/dev/nvidia0
```

## 9. What to do with the services at this point

Right after the packages are installed but before the final cluster configuration exists, do not try to start the whole cluster on the default configuration.

`mgmt01` ultimately runs:

```text
munge + slurmctld + slurmd
```

Every GPU worker ultimately runs:

```text
munge + slurmd
```

Enabling and restarting them is handled by the scripts below. **Note that `--apply` comes last in both, with the configuration file paths before it** — the opposite of the numbered scripts such as `10`/`20`/`25`:

```bash
# mgmt01
sudo ./scripts/cluster/install-controller-config.sh \
  config/slurm/slurm.conf.generated \
  --apply

# every GPU worker
sudo ./scripts/cluster/install-worker-config.sh \
  <local path to munge.key> \
  <local path to slurm.conf.generated> \
  --apply
```

Do not enable `slurmctld` on a worker.

## 10. Final verification

After the configuration is installed, run on each host:

```bash
sudo slurmd -C
sudo slurmd -G
systemctl --no-pager --full status munge slurmd
```

On `mgmt01` only:

```bash
systemctl --no-pager --full status slurmctld
scontrol ping
sinfo -N -l
```

The Slurm installation is complete only when the software version, the cgroup plugin, GRES, Munge and the per-node smoke tests all pass.
