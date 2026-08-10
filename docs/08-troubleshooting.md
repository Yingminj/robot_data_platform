# Installation and runtime troubleshooting

**English** | [简体中文](08-troubleshooting.zh-CN.md)

Locate the failure by layer first; do not reinstall every component the moment the web UI misbehaves:

```text
host/network
→ NFS, accounts, time
→ Munge
→ Slurm controller/worker
→ the shared training environment
→ leLab systemd
→ leLab SSH GPU probing
→ datasets and training jobs
```

## 0. These outputs are not failures

Before troubleshooting, rule out three outputs that are **normal but look like errors**. They have caused pointless reinstallations more than once:

| Output | When it appears | Explanation |
|---|---|---|
| `DNS SRV lookup failed` / `Could not establish a configuration source` | `sbatch --version` on a worker **before** the Slurm config is installed | newer Slurm loads a configuration before printing the version; this machine has no `/etc/slurm/slurm.conf` yet and falls back to the configless discovery this cluster does not use. It disappears once the config is installed. Use `/usr/sbin/slurmd -V` to check the version |
| `_normalize_sys_gres_types ... Setting system GRES type to NULL` | on every `slurmd -G` | `gres.conf` uses a model-free `Name=gpu` while NVML reports `nvidia_geforce_rtx_4090`, so the type is set to NULL, consistent with `Gres=gpu:1`. The next line, `Gres Name=gpu Type=(null) Count=1`, is the actual conclusion |
| `couldn't chdir to ...: going to /tmp instead` | `srun` from the repository directory on `mgmt01` | the submitting side's current directory does not exist on the worker. leLab jobs use absolute paths and are unaffected. Add `--chdir=/tmp` to silence it |

## 1. Collect the minimum state first

On `mgmt01`:

```bash
hostname -s
slurmd -V
stat -fc %T /sys/fs/cgroup
systemctl is-active munge slurmctld slurmd lelab-platform
scontrol ping
sinfo -N -l
squeue
findmnt /mnt/robot_platform
curl --noproxy '*' -fsS http://127.0.0.1:8000/cluster/status | \
  jq -c '.nodes[] | {name,reachable,slurm_state,eligible,reason}'
```

On **every** worker:

```bash
hostname -s
slurmd -V
stat -fc %T /sys/fs/cgroup
systemctl is-active munge slurmd
nvidia-smi
sudo slurmd -G
findmnt /mnt/robot_platform
sha256sum /etc/slurm/slurm.conf
```

The output of the last command must be identical to the one on `mgmt01`. **An inconsistent cluster-wide `slurm.conf` is the most common root cause after adding a node**, and the symptom (one node `DOWN`) does not point at the configuration.

## 2. The worker installation script only prints usage

The error:

```text
ERROR: usage: ./scripts/cluster/install-worker-config.sh \
<secure-copy-of-munge.key> <slurm.conf.generated> --apply
```

This does not mean `--apply` was line-wrapped wrongly; it means at least one of the following does not hold:

- the first file does not exist on this worker or is unreadable;
- the second file does not exist on this worker or is unreadable;
- the third argument is not `--apply`.

Check on `gpu01`:

```bash
sudo test -r /home/snorlax/robot-platform-secure/munge.key
sudo test -r /home/snorlax/robot-platform-secure/slurm.conf.generated
```

The correct invocation:

```bash
sudo ./scripts/cluster/install-worker-config.sh \
  /home/snorlax/robot-platform-secure/munge.key \
  /home/snorlax/robot-platform-secure/slurm.conf.generated \
  --apply
```

`/secure/temp/...` is a placeholder, not a directory the script creates.

## 2b. The controller installation script only prints usage

```text
This script changes the host. Re-run it with --apply after reviewing config/site.env.
```

**The argument order of `install-controller-config.sh` is the reverse of the numbered scripts**: the configuration file path is the first argument and `--apply` is the second. Passing only `--apply` makes it be read as the configuration filename, so the script prints the same message as with no arguments at all.

```bash
# wrong: --apply is taken as the configuration filename
sudo ./scripts/cluster/install-controller-config.sh --apply

# correct
sudo ./scripts/cluster/install-controller-config.sh \
  config/slurm/slurm.conf.generated \
  --apply
```

The same goes for `install-worker-config.sh`: the two file paths come first and `--apply` is third. The numbered scripts such as `10`/`20`/`25`, by contrast, put `--apply` first.

## 2c. The render script reports a node count mismatch

```text
expected 4 nodes, found 2
```

The number of `NodeName=` lines in `config/slurm/nodes.conf` does not match the node count in `GPU_NODE_NAMES` in `config/site.env`. When adding a node this usually means `site.env` was updated but the new line was not appended to `nodes.conf`.

Related errors:

| Error | Cause |
|---|---|
| `missing gpu02 in .../nodes.conf` | the line for that node is missing from `nodes.conf`, or the NodeName spelling differs from `GPU_NODE_NAMES` |
| `nodes.conf still contains FILL_ME placeholders` | the new node's hardware parameters have not been filled in; run `sudo slurmd -C` on that node first |
| `GPU_NODE_NAMES and GPU_NODE_IPS have different lengths` | the two lists have different element counts; they correspond positionally |

## 2d. Existing nodes go DOWN after adding a node

Slurm requires `slurm.conf` to be byte-for-byte identical cluster-wide. When only the new node gets the new configuration, the existing nodes still hold the old one without that node, and they fail once the controller reloads.

```bash
# compare on every node; all must match
sha256sum /etc/slurm/slurm.conf
```

The fix is to redistribute the newly rendered `slurm.conf.generated` to **all** workers (the Munge key is unchanged and does not need resending), then run `sudo systemctl restart slurmd` on each. The full flow is in [Adding a GPU node to an existing cluster](09-add-gpu-node.md).

## 2e. The managed /etc/hosts block refuses to update

```text
existing managed /etc/hosts block differs; review it manually
```

`05-configure-hosts.sh` **makes no incremental edits**: it compares the existing managed block with the target block and stops when they differ, to avoid overwriting manual adjustments. When the topology changes, delete the old block first and regenerate it, on **every** host:

```bash
sudo cp -a /etc/hosts /etc/hosts.bak.$(date +%Y%m%d%H%M%S)
sudo sed -i '/^# BEGIN robot-platform managed hosts$/,/^# END robot-platform managed hosts$/d' /etc/hosts
sudo ./scripts/05-configure-hosts.sh --apply
getent hosts mgmt01 gpu01 gpu02 gpu03
```

The other stop condition is an entry with the same name that already exists **outside** the block, reported as `/etc/hosts already contains gpu02`; that line has to be cleaned up by hand first.

## 3. Slurm cgroup v2 plugin errors

Check the version and the system first:

```bash
slurmd -V
stat -fc %T /sys/fs/cgroup
find /usr/lib -type f -name cgroup_v2.so -print
```

Currently expected:

```text
slurm 26.05.2
cgroup2fs
```

If the old version shipped with Ubuntu 22.04 is still in place, upgrade every machine following [Slurm 26.05.2 installation](Slurm-INSTALL.md). Upgrading only the controller, or only some workers, is not an option.

## 4. A Slurm node is DOWN, INVAL or UNKNOWN

On `mgmt01`:

```bash
scontrol show node <NodeName>
journalctl -u slurmctld -n 150 --no-pager
```

On the failing worker:

```bash
sudo slurmd -C
sudo slurmd -G
journalctl -u slurmd -u munge -n 150 --no-pager
```

Compare item by item:

1. NodeName against `hostname -s`;
2. `NodeAddr` against the static IP;
3. the CPU topology and `RealMemory` (**a declared value above what `slurmd -C` measured causes `INVAL`**; machines usually differ by a few MB, so do not copy values between them);
4. the `slurm.conf`, `cgroup.conf` and `gres.conf` checksums being identical on **all** nodes;
5. the Munge key checksum, `munge:munge` ownership and mode `0400`;
6. the time on each machine;
7. TCP 6817/6818;
8. whether `sudo slurmd -G` recognizes `gpu:1`.

Item 4 is the most common cause after adding a node — see section 2d.

Resume a node only once the cause is actually fixed:

```bash
sudo scontrol update NodeName=gpu02 State=RESUME
```

## 5. NFS is mounted but the service account cannot write

Check:

```bash
findmnt /mnt/robot_platform
sudo -u robot-train test -r /mnt/robot_platform/datasets
sudo -u robot-train test -w /mnt/robot_platform/jobs
sudo -u robot-ingest test -w /mnt/robot_platform/mlflow-artifacts
```

While the QNAP uses `all_squash`, a local Linux `chown` usually does not help, because every client account is mapped to the QNAP guest. On the QNAP:

- confirm the client IP is in the NFS allow list;
- confirm the share is RW;
- give the guest account read/write permission on the share and its subdirectories.

## 6. The leLab installation reports the Python packages succeeded, then EOF

Seen once:

```text
unexpected EOF while looking for matching `"'
```

Check the current repository script first:

```bash
bash -n scripts/15-install-lelab-platform.sh
```

Once the syntax check passes, run it again:

```bash
sudo ./scripts/15-install-lelab-platform.sh --apply
```

`Successfully installed LeLab ...` only proves the Python installation stage finished. Also confirm:

```bash
test -r /etc/robot-platform/lelab.env
test -r /etc/systemd/system/lelab-platform.service
systemctl is-active lelab-platform
```

## 7. curl against the local API returns 502

If `http_proxy` or `https_proxy` is set, `curl` may send even a `127.0.0.1` request to the proxy.

Check:

```bash
env | grep -Ei '^(http|https|all|no)_proxy='
```

When reaching local services, use:

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8000/health
```

When a command is wrapped across lines, there must be no space after the backslash `\`.

## 8. ssh-copy-id cannot open the private key

The error:

```text
failed to open ID file '/etc/robot-platform/lelab_ssh_key': Permission denied
```

The public key exists, but the ordinary user cannot read the private key of the same name. Ask explicitly for the public key only:

```bash
ssh-copy-id \
  -f \
  -i /etc/robot-platform/lelab_ssh_key.pub \
  snorlax@192.168.100.215
```

The private key must stay:

```text
robot-train:robotdata 0600
```

Do not loosen the private key permissions just to quiet `ssh-copy-id`.

## 9. leLab reports Host key verification failed

This has nothing to do with `Permission denied`: **what is missing is the known_hosts entry for `robot-train` on `mgmt01`**, not the remote public key. Configure known_hosts for the service account that actually initiates the connection; the full fingerprint verification steps are in [leLab host fingerprint configuration](07-lelab-cluster-web.md#5-verify-and-install-the-worker-host-fingerprints).

Key points: scan one node at a time into a separate file, **do not add `-H`** (a hashed line cannot be matched against the fingerprint seen on the node, nor deduplicated), compare only the `SHA256:` field, and once verified, **append** with `sudo -u robot-train tee -a` rather than overwriting.

Verification must use the same identity as systemd:

```bash
sudo -H -u robot-train ssh \
  -o BatchMode=yes \
  -i /etc/robot-platform/lelab_ssh_key \
  snorlax@192.168.100.215 \
  nvidia-smi -L
```

Your own user being able to SSH does not mean `robot-train` can.

## 9b. leLab reports Permission denied (publickey,password)

```json
{"name":"gpu02","reachable":false,"reason":"yang@192.168.100.216: Permission denied (publickey,password)."}
```

**This error means the host key already passed** — leave known_hosts alone. What is missing is the leLab public key in the remote `authorized_keys`.

On `mgmt01`:

```bash
cat /etc/robot-platform/lelab_ssh_key.pub
ssh-copy-id -f -i /etc/robot-platform/lelab_ssh_key.pub yang@192.168.100.216
```

`-f` cannot be omitted, otherwise `ssh-copy-id` tries to read the private key of the same name, which only `robot-train` can read. When pasting by hand, make sure the public key does not wrap, and verify with `ssh-keygen -lf ~/.ssh/authorized_keys`.

Once installed, leLab does not need restarting — `/cluster/status` probes live on every request.

The difference from `Host key verification failed`:

| Error | What is missing |
|---|---|
| `Permission denied (publickey,password)` | the public key in the remote `authorized_keys` |
| `Host key verification failed` | the known_hosts entry for `robot-train` on `mgmt01` |

## 10. A node is reachable but eligible is false

Example:

```json
{
  "slurm_state": "idle",
  "reachable": true,
  "compute_processes": 1,
  "eligible": false,
  "reason": "GPU has a compute process outside or inside Slurm"
}
```

This means SSH and GPU probing already succeeded, but there is a CUDA compute process on the GPU. Since Slurm also reports `idle`, it is usually a manual process outside Slurm.

To investigate:

```bash
sudo -H -u robot-train ssh \
  -o BatchMode=yes \
  -i /etc/robot-platform/lelab_ssh_key \
  snorlax@192.168.100.215 \
  nvidia-smi \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits
```

Inspect the PID on that node:

```bash
ps -o user,pid,ppid,lstart,cmd -p <PID>
```

Do not kill an unknown process. Once confirmed to be a stale job, have its owner stop it. After the process disappears, request the API again — leLab does not need restarting.

If the occupant is a remote desktop tool (RustDesk, TeamViewer, Sunlogin and similar), the right fix is not to close it but to add the process name to the `graphics_patterns` allow list in `_probe_node` in `apps/lelab/lelab/cluster.py`. `rustdesk` is already there, which is why `compute_processes` is 0 on `mgmt01` even while it runs.

### Relationship to scheduling

A node with `eligible: false` **will not be picked by `auto`**, but it can still be selected manually in the UI, and Slurm still accepts `srun`/`sbatch` for it as usual. So `sinfo` reporting `idle` while leLab reports unavailable is not a contradiction — the former looks at Slurm allocation state, the latter additionally looks at whether a non-Slurm process is on the GPU.

## 11. Confusing the SSH address with the Slurm node name

Correct:

```bash
LELAB_CLUSTER_NODES=mgmt01=192.168.100.202,gpu01=snorlax@192.168.100.215,gpu02=yang@192.168.100.216,gpu03=snorlax@192.168.100.217
```

The left side is the Slurm NodeName, the right side is the SSH target. Note that `gpu02` uses `yang`, unlike the other two — **the SSH user does not have to be the same on every node**. The following are all wrong:

```text
changing the Slurm NodeName into snorlax@192.168.100.215
writing the gpu02 mapping in /etc/hosts as a string that includes a user
assuming the SSH user must equal the NodeName
assuming every node uses the same SSH user
```

## 11b. Two kinds of accident caused by pasting long commands

Pasting a long command into a terminal inserts a newline at an arbitrary position. This has caused two real accidents in this project, and both are worth remembering individually.

### A truncated sed expression

```text
sed: -e expression #1, char 85: unterminated `s' command
```

For a very long single-line value such as `LELAB_CLUSTER_NODES`, **edit it in an editor, not with a one-line `sed`**. When the command line is unavoidable, assemble it from short fragments and `echo` it first to confirm it is one complete line:

```bash
N='mgmt01=192.168.100.202'
N="$N,gpu01=snorlax@192.168.100.215"
N="$N,gpu02=yang@192.168.100.216"
N="$N,gpu03=snorlax@192.168.100.217"
echo "$N"
sudo sed -i "s|^LELAB_CLUSTER_NODES=.*|LELAB_CLUSTER_NODES=$N|" /etc/robot-platform/lelab.env
```

The `sed` script must use double quotes here, because `$N` has to expand.

The same class of problem hits SSH public keys: a wrapped public key in `authorized_keys` **can never match, with no error at all**, and the symptom is `Permission denied (publickey)`. Use `ssh-keygen -lf ~/.ssh/authorized_keys` to confirm every line parses.

### An unset variable writing into the root directory

When distributing the Munge key, if only the second half is pasted and `stage_dir="$(mktemp -d)"` is missed:

```bash
sudo install -o "$USER" -g "$(id -gn)" -m 0600 /etc/munge/munge.key "$stage_dir/munge.key"
```

`$stage_dir` is empty, the path becomes `/munge.key`, and because this line runs under `sudo` it **silently succeeds**, writing the cluster's unique authentication key into the filesystem root. Only the next line, which has no `sudo`, reports `install: cannot create regular file '/slurm.conf.generated': Permission denied` — the error comes from the second line while the damage was done by the first.

Check and destroy it:

```bash
ls -l /munge.key && sudo shred -u /munge.key
```

The distribution commands in this documentation are always written as `"${stage_dir:?}/munge.key"`. The `:?` makes bash report `stage_dir: parameter null or not set` and abort immediately when the variable is empty — do not remove it.

## 12. Slurm reports it cannot enter the submission directory on the remote side

```text
error: couldn't chdir to `/home/kewei/YING/robot_data_platform': No such file or directory: going to /tmp instead
```

**This is a warning, not a failure.** `srun`/`sbatch` inherit the submitting side's current directory by default, and that repository path only exists on `mgmt01`.

For an ad hoc smoke test, name a directory that exists on every worker:

```bash
srun --chdir=/tmp <other arguments> <command>
```

Real leLab jobs use absolute paths for the script, the logs and the output:

```text
/mnt/robot_platform/jobs/<job-id>
```

That absolute path must be the same on **every** worker. If training really fails, check `job.sbatch`, `slurm.out` and `scontrol show job <id>`; do not diagnose the failure from the chdir warning alone.

Jobs submitted by leLab now carry `--chdir=/mnt/robot_platform/jobs/<job-id>` and no longer print this warning; the service's `WorkingDirectory=/opt/robot-platform/lelab` only exists on `mgmt01`. If you still see it, the service has not been restarted onto the new version.

## 12b. A job is submitted successfully but fails immediately on one node

A job that fails right after being scheduled onto a newly added node while working fine on the older ones usually means that node is missing a runtime directory. This class of problem **raises no error at submission time** and only surfaces once the job lands on that node.

Check on that node:

```bash
sudo -u robot-train test -w /var/lib/robot-platform/cache && echo cache OK
findmnt /mnt/robot_platform
sudo -u robot-train test -w /mnt/robot_platform/jobs && echo jobs OK
sudo -u robot-train /opt/robot-platform/train-venv/bin/python -c \
  'import torch, lerobot; print(torch.cuda.is_available())'
sudo -u robot-train /opt/robot-platform/train-venv/bin/python -c \
  'from torchcodec.decoders import VideoDecoder; print("decoder OK")'
```

For a failure on the last one, see 12b-2: what is missing is the system `ffmpeg` package, not a Python dependency.

`/var/lib/robot-platform/cache` is `LELAB_JOB_CACHE_ROOT`. Slurm points `HOME` at the `robot-train` home directory, which does not exist on the workers, so a job that caches into `~` fails. Create it:

```bash
sudo install -d -o robot-train -g robotdata -m 0750 /var/lib/robot-platform/cache
```

When the NAS is not mounted or is read-only, first confirm the QNAP NFS allow list contains the IP of **this** node.

### 12b-1. `Cannot create the job cache under '/var/lib/robot-platform/cache'`, when that directory clearly exists and is writable

The error message points at `LELAB_JOB_CACHE_ROOT`, but what actually failed is the different path printed by the `mkdir` on the line above:

```text
mkdir: cannot create directory '/var/lib/robot-platform/huggingface': Permission denied
Cannot create the job cache under '/var/lib/robot-platform/cache' on gpu03;
```

`sbatch` defaults to `--export=ALL`, so the job inherits the leLab service's own `HF_HOME` (`/var/lib/robot-platform/huggingface` from `/etc/robot-platform/lelab.env`). That is a cache path **local to the management node**; it does not exist on a new worker, and its parent `/var/lib/robot-platform` is owned by root so `robot-train` cannot create it — the whole `mkdir -p` therefore fails. The `cache` directory itself is fine, and `torch`, `xdg` and `home` under it have already been created.

**Read the specific path in the `mkdir` line, not just the summary on the second line.**

The fix is already in `apps/lelab/lelab/runners/slurm.py`: once `LELAB_JOB_CACHE_ROOT` is set it overrides the inherited `HF_HOME`, so all caches land under `$LELAB_JOB_CACHE_ROOT/`. Restart the service to pick it up:

```bash
sudo systemctl restart lelab-platform
```

If the service cannot be restarted right now, creating the directory on that node also works around it:

```bash
sudo install -d -o robot-train -g robotdata -m 0750 /var/lib/robot-platform/huggingface
```

### 12b-2. `RuntimeError: Could not load libtorchcodec` right after training starts

The log has already printed `Start offline training on a fixed dataset`, `num_learnable_params` and similar lines, so the configuration, the dataset and the GPU are all fine; then the first batch throws inside a DataLoader worker:

```text
RuntimeError: Caught RuntimeError in DataLoader worker process 0.
  ...
RuntimeError: Could not load libtorchcodec.
[start of libtorchcodec loading traceback]
FFmpeg version 8:
OSError: libavutil.so.60: cannot open shared object file: No such file or directory
...
FFmpeg version 4:
OSError: libavdevice.so.58: cannot open shared object file: No such file or directory
```

torchcodec tries FFmpeg versions 4, 5, 6, 7 and 8 in turn. **Ubuntu 22.04 only has FFmpeg 4.4, so only the last block, `FFmpeg version 4`, is the real cause; the first four are bound to fail and can be ignored.** The incompatible-PyTorch-version suggestion in the error (item 2) is usually misleading too.

The real cause is that the node is missing the system `ffmpeg` package. torchcodec ships its own `libtorchcodec_core*.so`, but the `libav*.so` libraries it depends on come from the system, and no pip dependency installs them. The tricky part is that other packages incidentally pull in `libavcodec58`, `libavformat58` and `libavutil56`, while `libavdevice58` is provided only by `ffmpeg` — so the node looks like it "has the ffmpeg libraries" while missing exactly that one.

Confirm it (on the failing node, compared against a healthy one):

```bash
ls /usr/lib/x86_64-linux-gnu/libavdevice.so.58   # missing means this is the problem
dpkg -l ffmpeg
```

Fix:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
sudo -u robot-train /opt/robot-platform/train-venv/bin/python -c \
  'from torchcodec.decoders import VideoDecoder; print("decoder OK")'
```

`25-install-training-environment.sh` now includes `ffmpeg` in its package list and performs this import check at the end, and `90-validate-deployment.sh` has gained a `video decoder loads FFmpeg` item; nodes installed before that change need it added by hand.

## 13. Log locations

```bash
# leLab
journalctl -u lelab-platform -n 200 --no-pager

# Slurm controller
journalctl -u slurmctld -n 200 --no-pager

# workers and authentication
journalctl -u slurmd -u munge -n 200 --no-pager

# management infrastructure
sudo docker compose \
  --env-file deploy/management/.env \
  -f deploy/management/compose.yaml \
  ps
```

When troubleshooting, keep the command output, timestamps, node names and job IDs; never send keys, database passwords or tokens.
