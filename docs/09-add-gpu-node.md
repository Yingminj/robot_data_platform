# Adding a GPU node to an existing cluster

**English** | [简体中文](09-add-gpu-node.zh-CN.md)

This document covers adding a GPU worker to an **already running cluster**, for example growing from `mgmt01 + gpu01` to `mgmt01 + gpu01 + gpu02 + gpu03`.

For a first deployment, do not use this document — follow the complete order in the [README](../README.md).

## 0. The three easy mistakes this document exists for

A first scale-out most often fails at these three points, and in each case the symptom does not point at the real cause:

1. **The topology fields must be updated on all hosts at the same time.** If `GPU_NODE_NAMES` is only changed on the new machine, both `05-configure-hosts.sh` and the render script still pass, but the cluster behaves inconsistently.
2. **`slurm.conf` must be identical cluster-wide.** If only the new node receives the new configuration, **the existing `gpu01` fails because it still holds the old configuration without the new node**. This is the step most often missed.
3. **The managed `/etc/hosts` block cannot be edited incrementally.** When `05-configure-hosts.sh` finds the existing block differs from the target, it stops outright; the old block has to be deleted by hand first.

The full order (do not skip steps):

```text
1. Install the OS, driver and Python 3.12 on the new node
2. Update GPU_NODE_NAMES / GPU_NODE_IPS in site.env on all hosts
3. Rebuild the managed /etc/hosts block on all hosts
4. Run 20 + 25 on the new node and install Slurm 26.05.2
5. Run slurmd -C on the new node and fill the result into nodes.conf on mgmt01
6. Render the new slurm.conf.generated on mgmt01
7. Install the controller config on mgmt01
8. Distribute to *all* workers, including the existing gpu01
9. Verify every node is idle in sinfo
10. Configure leLab's SSH probing and node mapping
11. Acceptance
```

## 1. Determine the new node information

This document uses two new nodes as the example:

| Slurm NodeName | IP | SSH login |
|---|---|---|
| `gpu02` | `192.168.100.216` | `yang` |
| `gpu03` | `192.168.100.217` | `snorlax` |

**The SSH account does not have to match the Slurm NodeName, and does not have to be the same across nodes.** In the table above `gpu02` uses `yang` and `gpu03` uses `snorlax`, which is allowed: Slurm only knows the NodeName, and leLab's `LELAB_CLUSTER_NODES` maps NodeNames to SSH targets.

On the new node:

```bash
sudo hostnamectl set-hostname gpu02
```

Log in again afterwards and confirm `hostname -s` exactly matches the NodeName that will be written into `nodes.conf`.

## 2. Synchronize site.env on all hosts

**This step must be run separately on `mgmt01`, on `gpu01` and on every new node**, changing them to exactly the same values:

```bash
editor config/site.env
```

```bash
GPU_NODE_NAMES="mgmt01 gpu01 gpu02 gpu03"
GPU_NODE_IPS="192.168.100.202 192.168.100.215 192.168.100.216 192.168.100.217"
```

The two lists correspond positionally and must have the same length, or `05-configure-hosts.sh` reports `GPU_NODE_NAMES and GPU_NODE_IPS have different lengths`.

If the new node has a fresh copy of the repository, run `cp config/site.env.example config/site.env` first, then check it against [the list of fields that must match in the README](../README.md#0-prepare-the-shared-configuration). `DATA_GID` and `TRAIN_UID` matter especially: mismatched numbers mean jobs cannot write to the NAS from the new node.

## 3. Rebuild the managed /etc/hosts block

`05-configure-hosts.sh` **does not edit an existing managed block incrementally**. It compares the existing block with the target block and stops when they differ:

```text
existing managed /etc/hosts block differs; review it manually
```

This is deliberate, to keep the script from overwriting manual adjustments. The right approach when scaling out is to delete the old block and regenerate it. On **every** host:

```bash
sudo cp -a /etc/hosts /etc/hosts.bak.$(date +%Y%m%d%H%M%S)
sudo sed -i '/^# BEGIN robot-platform managed hosts$/,/^# END robot-platform managed hosts$/d' /etc/hosts
sudo ./scripts/05-configure-hosts.sh --apply
getent hosts mgmt01 gpu01 gpu02 gpu03
```

All four lines must resolve in the last command.

## 4. Install the base components and training environment on the new node

On the new node, exactly as in [GPU node installation](03-gpu-node.md):

```bash
./scripts/00-audit-host.sh gpu
sudo ./scripts/20-install-gpu-node.sh --apply
sudo ./scripts/25-install-training-environment.sh --apply
```

Then install **exactly the same** package version as the existing nodes, following [Slurm 26.05.2 installation](Slurm-INSTALL.md):

```bash
/usr/sbin/slurmd -V     # expect slurm 26.05.2
stat -fc %T /sys/fs/cgroup   # expect cgroup2fs
```

At this point `sbatch --version` reports an error, **which is normal**:

```text
sbatch: error: resolve_ctls_from_dns_srv: res_nsearch error: Unknown host
sbatch: error: fetch_config: DNS SRV lookup failed
sbatch: fatal: Could not establish a configuration source
```

Newer Slurm tries to load a configuration before printing the version; this machine has no `/etc/slurm/slurm.conf` yet, so it falls back to the configless DNS SRV discovery this cluster does not use. The error disappears once the configuration is installed in step 8. Checking the version with `/usr/sbin/slurmd -V` is unaffected.

## 5. Collect the new node's hardware parameters

On **every new node**:

```bash
sudo slurmd -C
```

Copy `CPUs`, `Boards`, `SocketsPerBoard`, `CoresPerSocket`, `ThreadsPerCore` and `RealMemory` from the output into `config/slurm/nodes.conf` on `mgmt01`, **appending** new lines without touching the existing ones:

```ini
NodeName=gpu02 NodeAddr=192.168.100.216 CPUs=32 Boards=1 SocketsPerBoard=1 CoresPerSocket=16 ThreadsPerCore=2 RealMemory=61919 Gres=gpu:1 State=UNKNOWN
NodeName=gpu03 NodeAddr=192.168.100.217 CPUs=32 Boards=1 SocketsPerBoard=1 CoresPerSocket=16 ThreadsPerCore=2 RealMemory=61914 Gres=gpu:1 State=UNKNOWN
```

`RealMemory` usually differs by a few MB between machines; **do not copy one machine's value to another for the sake of tidiness**. A declared value above what `slurmd -C` measured puts the node into `INVAL`.

The number of lines in `nodes.conf` must equal the node count in `GPU_NODE_NAMES`; the render script checks this:

```text
expected 4 nodes, found 2
```

## 6. Render and review the configuration

On `mgmt01`:

```bash
./scripts/cluster/render-slurm-config.sh
grep -c '^NodeName=' config/slurm/slurm.conf.generated    # expect 4
grep '^PartitionName=' config/slurm/slurm.conf.generated
```

All three partitions should list every node:

```ini
PartitionName=debug Nodes=mgmt01,gpu01,gpu02,gpu03 Default=YES MaxTime=01:00:00 State=UP
PartitionName=train Nodes=mgmt01,gpu01,gpu02,gpu03 MaxTime=7-00:00:00 State=UP
PartitionName=eval Nodes=mgmt01,gpu01,gpu02,gpu03 MaxTime=1-00:00:00 State=UP
```

The render script fails outright on a `FILL_ME`, which prevents deploying before the hardware parameters are filled in.

## 7. Install the controller configuration

On `mgmt01`:

```bash
sudo ./scripts/cluster/install-controller-config.sh \
  config/slurm/slurm.conf.generated \
  --apply
```

**The configuration file path is the first argument and `--apply` is the second.** This differs from the numbered scripts (`10`/`20`/`25` and friends put `--apply` first). Writing only `sudo ./scripts/cluster/install-controller-config.sh --apply` makes `--apply` be read as the configuration filename, and it prints the same message:

```text
This script changes the host. Re-run it with --apply after reviewing config/site.env.
```

The script restarts `slurmctld` **and the local `slurmd` on mgmt01**. Check `squeue` before running it: jobs running on this machine will be interrupted.

## 8. Distribute to all workers (including the existing ones)

> **This is the step where the existing `gpu01` is most often forgotten.** Slurm requires `slurm.conf` to be identical cluster-wide. `gpu01` still holds the old two-node configuration and fails once the controller reloads.

The Munge key is unchanged for existing nodes, so only the **new nodes** need the key; `slurm.conf`, on the other hand, must be updated on **all** workers.

Stage and distribute from `mgmt01`:

```bash
stage_dir="$(mktemp -d)"
sudo install -o "$USER" -g "$(id -gn)" -m 0600 \
  /etc/munge/munge.key "${stage_dir:?}/munge.key"
install -m 0644 \
  config/slurm/slurm.conf.generated "${stage_dir:?}/slurm.conf.generated"

for target in snorlax@192.168.100.215 yang@192.168.100.216 snorlax@192.168.100.217; do
  ssh "$target" 'install -d -m 0700 ~/robot-platform-secure'
  scp "${stage_dir:?}"/munge.key "${stage_dir:?}"/slurm.conf.generated \
    "$target:~/robot-platform-secure/"
done

shred -u "${stage_dir:?}/munge.key"
rm -f "${stage_dir:?}/slurm.conf.generated"
rmdir "${stage_dir:?}"
```

The `:?` in `${stage_dir:?}` must not be dropped. If only the second half of the commands is copy-pasted and `stage_dir="$(mktemp -d)"` is missed, the variable is empty, `"$stage_dir/munge.key"` becomes `/munge.key`, and the earlier `sudo` line **silently succeeds**, writing the cluster's unique authentication key into the filesystem root. With `:?`, bash immediately reports `stage_dir: parameter null or not set` and aborts.

If it already happened, delete it:

```bash
sudo shred -u /munge.key
```

Install on **every** worker:

```bash
sudo ./scripts/cluster/install-worker-config.sh \
  ~/robot-platform-secure/munge.key \
  ~/robot-platform-secure/slurm.conf.generated \
  --apply
```

Clean up afterwards:

```bash
shred -u ~/robot-platform-secure/munge.key
rm -f ~/robot-platform-secure/slurm.conf.generated
rmdir ~/robot-platform-secure
```

`slurmd -G` prints a GRES type notice, **which is normal**:

```text
gres/gpu: _normalize_sys_gres_types: Could not find an unused configuration record
with a GRES type that is a substring of system device `nvidia_geforce_rtx_4090`.
Setting system GRES type to NULL
```

`gres.conf` declares `Name=gpu` with no model, NVML reports the device model as `nvidia_geforce_rtx_4090`, so Slurm sets the type to NULL. That is consistent with the equally model-free `Gres=gpu:1` in `nodes.conf`. The line right after it is the actual conclusion:

```text
Gres Name=gpu Type=(null) Count=1 Index=0 File=/dev/nvidia0 Flags=HAS_FILE,ENV_NVML
```

`gres.conf` only needs changing if you want to request GPUs by model (`--gres=gpu:rtx4090:1`).

## 9. Verify cluster-wide consistency

Run on every host; the four outputs must match across machines:

```bash
sha256sum /etc/slurm/slurm.conf /etc/slurm/cgroup.conf /etc/slurm/gres.conf
sudo sha256sum /etc/munge/munge.key
```

**Never paste the contents of the Munge key anywhere; compare checksums only.**

On `mgmt01`:

```bash
scontrol ping
sinfo -N -l
```

Every node is expected to be `idle` in all three partitions `debug`, `train` and `eval`.

Per-node smoke test:

```bash
for node in mgmt01 gpu01 gpu02 gpu03; do
  echo "[$node]"
  srun --partition=debug --nodelist="$node" --gres=gpu:1 --time=00:02:00 \
    nvidia-smi -L
done
```

Each must return a **different** GPU UUID. A repeated UUID means `NodeAddr` is wrong and two NodeNames point at the same physical machine.

This warning during the run can be ignored:

```text
error: couldn't chdir to `/home/kewei/YING/robot_data_platform': No such file or directory: going to /tmp instead
```

`srun` passes the submitting side's current directory to the remote end, and that path only exists on `mgmt01`. Jobs submitted by leLab use absolute paths and are unaffected. Add `--chdir=/tmp` to silence the warning.

## 10. Configure leLab

leLab is installed on `mgmt01` only and does not need reinstalling for a scale-out — just these three things.

### 10.1 The SSH public key

Install the leLab public key from `mgmt01` into the login account on each new node:

```bash
# view it on mgmt01
cat /etc/robot-platform/lelab_ssh_key.pub
```

On the new node, as the matching account (`yang` for `gpu02`, `snorlax` for `gpu03`):

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
editor ~/.ssh/authorized_keys      # paste the line from above
chmod 600 ~/.ssh/authorized_keys
ssh-keygen -lf ~/.ssh/authorized_keys
```

The public key is one very long line. **If it wraps when pasted into the terminal, you get a key that can never match, with no error at all.** Use the last command to confirm the new key parses correctly.

`ssh-copy-id` works too; note that `-f` must be kept (otherwise it tries to read the private key of the same name, which only `robot-train` can read):

```bash
ssh-copy-id -f -i /etc/robot-platform/lelab_ssh_key.pub yang@192.168.100.216
```

### 10.2 The host key

The connection is initiated by `robot-train` on `mgmt01`, so the entry goes into its `known_hosts`. **Scan one node at a time and verify the fingerprint manually; do not use `-H`**: with `-H` the hostname in the output is hashed, so it cannot be matched against the fingerprint seen on the node, nor deduplicated.

Scan on `mgmt01`:

```bash
ssh-keyscan -t ed25519 192.168.100.216 2>/dev/null > /tmp/kh216
ssh-keyscan -t ed25519 192.168.100.217 2>/dev/null > /tmp/kh217
ssh-keygen -lf /tmp/kh216
ssh-keygen -lf /tmp/kh217
```

Read the real fingerprint **on each new node's console**:

```bash
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
```

**Compare only the `SHA256:` field.** The trailing comment will always differ: the scan result labels the queried IP, while the node itself labels the key comment (such as `root@gpu02`). If the fingerprints differ, stop — whatever answers at that IP is not the machine you think it is.

Once verified, append (**do not overwrite** the existing content; the `gpu01` entry is still in there):

```bash
sudo -u robot-train tee -a /home/robot-train/.ssh/known_hosts < /tmp/kh216 >/dev/null
sudo -u robot-train tee -a /home/robot-train/.ssh/known_hosts < /tmp/kh217 >/dev/null
rm -f /tmp/kh216 /tmp/kh217
```

Use `sudo -u robot-train tee -a` here rather than `sudo ... >>`: the redirection is performed by your current shell under your own identity rather than `robot-train`, producing a file with the wrong owner that SSH then ignores outright.

Confirm it was written:

```bash
sudo -u robot-train ssh-keygen -F 192.168.100.216 -f /home/robot-train/.ssh/known_hosts
```

### 10.3 The node mapping

Edit the active configuration (**not** `config/lelab.env.example` in the repository):

```bash
sudo editor /etc/robot-platform/lelab.env
```

```bash
LELAB_CLUSTER_NODES=mgmt01=192.168.100.202,gpu01=snorlax@192.168.100.215,gpu02=yang@192.168.100.216,gpu03=snorlax@192.168.100.217
```

This is a very long line: **edit it in an editor, not with a one-line `sed`**. Pasting a long command into a terminal inserts a newline at an arbitrary position, and `sed` then sees an incomplete expression and errors:

```text
sed: -e expression #1, char 85: unterminated `s' command
```

When the command line is unavoidable, assemble the value from short fragments, none of which will wrap:

```bash
N='mgmt01=192.168.100.202'
N="$N,gpu01=snorlax@192.168.100.215"
N="$N,gpu02=yang@192.168.100.216"
N="$N,gpu03=snorlax@192.168.100.217"
echo "$N"
sudo sed -i "s|^LELAB_CLUSTER_NODES=.*|LELAB_CLUSTER_NODES=$N|" /etc/robot-platform/lelab.env
```

Before running `sed`, check that `echo "$N"` prints one complete line. Note that the `sed` script must use double quotes, because `$N` has to expand.

The file is read by systemd as an `EnvironmentFile` and **only takes effect on a service restart**:

```bash
sudo systemctl restart lelab-platform
```

## 11. Runtime prerequisites on the new node

None of these raise an error before a job is submitted; they only fail once a job is scheduled onto the new node. Confirm on **every new node**:

```bash
# LELAB_JOB_CACHE_ROOT must exist and be writable by robot-train.
# Slurm points HOME at the robot-train home directory, which does not exist on workers.
sudo install -d -o robot-train -g robotdata -m 0750 /var/lib/robot-platform/cache
sudo -u robot-train test -w /var/lib/robot-platform/cache && echo cache OK

# The NAS must be mounted at the same absolute path
findmnt /mnt/robot_platform
sudo -u robot-train test -r /mnt/robot_platform/datasets && echo datasets OK
sudo -u robot-train test -w /mnt/robot_platform/jobs && echo jobs OK

# The training environment
sudo -u robot-train /opt/robot-platform/train-venv/bin/python -c \
  'import torch, lerobot; print(torch.cuda.is_available())'

# The video decoding backend. torchcodec ships libtorchcodec_core*.so, but libav* comes
# from the system ffmpeg and no pip dependency installs it. When it is missing, every
# check above passes and the failure only appears at the first training batch as
# "Could not load libtorchcodec" (see [troubleshooting 12b-2](08-troubleshooting.md)).
sudo -u robot-train /opt/robot-platform/train-venv/bin/python -c \
  'from torchcodec.decoders import VideoDecoder; print("decoder OK")'
```

`25-install-training-environment.sh` now installs `ffmpeg` and performs this import check at the end of the installation, but **nodes installed before that change need it added by hand**:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

Also confirm the QNAP NFS allow list already contains the new node's IP, or the mount will fail or be read-only.

## 12. Acceptance

On `mgmt01`:

```bash
sinfo -N -l
curl --noproxy '*' -fsS http://127.0.0.1:8000/cluster/status | \
  jq -c '.nodes[] | {name,reachable,slurm_state,eligible,memory_free_mb,reason}'
```

Every new node should look like:

```json
{"name":"gpu02","reachable":true,"slurm_state":"idle","eligible":true,"memory_free_mb":47752,"reason":null}
```

Two common incomplete states:

| `reason` | Meaning |
|---|---|
| `Permission denied (publickey,password)` | the host key passed; the public key is not in the remote `authorized_keys` (see 10.1) |
| `Host key verification failed` | nothing to do with the public key; `robot-train`'s known_hosts lacks that node (see 10.2) |
| `GPU has a compute process outside or inside Slurm` | there is a CUDA process outside Slurm on that node's GPU, see [troubleshooting 10](08-troubleshooting.md) |

Finally, from the leLab UI, set the node selection to `auto`, submit a very short training job, and confirm it lands on the new node and produces `slurm.out` and a checkpoint.

## 13. Files that must be updated together

What to check after a scale-out:

| File | Location | In Git |
|---|---|---|
| `config/site.env` | all hosts | no, ignored |
| `config/slurm/nodes.conf` | `mgmt01` only | no, ignored |
| `config/slurm/slurm.conf.generated` | `mgmt01` only, generated by a script | no, ignored |
| `/etc/slurm/slurm.conf` | all hosts, must be identical | no |
| the managed `/etc/hosts` block | all hosts | no |
| `/etc/robot-platform/lelab.env` | `mgmt01` only | no |
| `/home/robot-train/.ssh/known_hosts` | `mgmt01` only | no |
| `config/site.env.example` | repository | yes |
| `config/lelab.env.example` | repository | yes |
| `config/slurm/nodes.conf.example` | repository | yes |
| the topology table in `README.md` | repository | yes |

The first seven are local active configuration and **do not come back on their own after a machine swap or a reinstall**; the only lasting record of a scale-out is the three `.example` templates and the README.
