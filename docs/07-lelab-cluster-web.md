# leLab cluster web installation and configuration

**English** | [简体中文](07-lelab-cluster-web.zh-CN.md)

leLab is installed on `mgmt01` only. It submits training through the local Slurm commands, and SSHes into every worker to run a read-only `nvidia-smi` so it can spot CUDA processes outside Slurm.

**Adding a node does not require reinstalling leLab** — only the three things in sections 3, 4 and 5: the node mapping, the SSH public key and the host key. The full scale-out flow is in [Adding a GPU node to an existing cluster](09-add-gpu-node.md).

```text
browser
  → mgmt01:8000 / leLab FastAPI
  → scan /mnt/robot_platform/datasets
  → sinfo for Slurm node state
  → local/SSH nvidia-smi for GPUs and compute processes
  → sbatch --nodes=1 --gres=gpu:1
  → /mnt/robot_platform/jobs/<job-id>
```

## 1. Installation prerequisites

Check on `mgmt01`:

```bash
python3.12 --version
node --version
npm --version
sbatch --version
sinfo -N -l
sudo -u robot-train test -r /mnt/robot_platform/datasets
sudo -u robot-train test -w /mnt/robot_platform/jobs
bash -n scripts/15-install-lelab-platform.sh
```

Requirements:

- Python 3.12;
- Node.js 20.19 or newer and npm;
- **all** nodes already `idle` in Slurm;
- `/opt/robot-platform/train-venv` installed on **every** worker;
- the NAS dataset directory readable and the jobs directory writable.

Node/npm must be available in the environment of the ordinary user who invokes sudo. The installer drops privileges to that user and builds the frontend in a temporary directory; root does not need its own Node installation.

## 2. Installation

Run on `mgmt01` only:

```bash
sudo ./scripts/15-install-lelab-platform.sh --apply
```

If you run it directly from a root shell or from an automation system, name the non-root user that owns Node/npm explicitly:

```bash
sudo LELAB_BUILD_USER=kewei \
  ./scripts/15-install-lelab-platform.sh --apply
```

Installation locations:

| Content | Path |
|---|---|
| Application | `/opt/robot-platform/lelab` |
| Python venv | `/opt/robot-platform/lelab-venv` |
| Runtime configuration | `/etc/robot-platform/lelab.env` |
| Model templates | `/etc/robot-platform/model-templates.json` |
| systemd service | `/etc/systemd/system/lelab-platform.service` |

The installer copies `lelab.env` and the model templates only when the files do not exist. Re-running the installation never overwrites an existing runtime configuration; to change the templates or the node list, edit the active files under `/etc/robot-platform` directly.

Check:

```bash
systemctl is-active lelab-platform
journalctl -u lelab-platform -n 100 --no-pager
curl --noproxy '*' -fsS http://127.0.0.1:8000/health
```

If the installation once failed at the end with a shell quoting or EOF error, first confirm the current script passes a syntax check, then re-run the same installation command. Python packages already being installed does not mean systemd and `/etc/robot-platform` are done.

## 3. Configure the node mapping

Edit the **active** configuration (not `config/lelab.env.example` in the repository, which is only a template):

```bash
sudo editor /etc/robot-platform/lelab.env
```

It should currently contain at least:

```bash
LELAB_CLUSTER_ENABLED=1
LELAB_CLUSTER_NODES=mgmt01=192.168.100.202,gpu01=snorlax@192.168.100.215,gpu02=yang@192.168.100.216,gpu03=snorlax@192.168.100.217
LELAB_SSH_CONNECT_TIMEOUT=3
LELAB_SSH_IDENTITY_FILE=/etc/robot-platform/lelab_ssh_key

LELAB_NAS_DATASETS_ROOT=/mnt/robot_platform/datasets
LELAB_OUTPUT_ROOT=/mnt/robot_platform/jobs
LELAB_MODEL_TEMPLATES=/etc/robot-platform/model-templates.json
HF_HOME=/var/lib/robot-platform/huggingface
LELAB_JOB_CACHE_ROOT=/var/lib/robot-platform/cache
```

The format of each node entry is:

```text
SlurmNodeName=SSHTarget
```

so:

- the left side must match the NodeName in `sinfo`;
- `mgmt01` is recognized as the local machine by node name and is not actually SSHed into;
- the right side may include an SSH user;
- **the SSH user does not have to be the same on every node** — in the example above `gpu02` uses `yang` while the other two use `snorlax`;
- do not change the Slurm NodeName into `snorlax@...`.

There is no upper bound on the node count; leLab splits this variable on commas and probes at most 8 nodes concurrently. Adding or removing a node means editing this one line — no code change and no reinstallation.

> **`LELAB_CLUSTER_NODES` is a very long line: edit it in an editor, not with a one-line `sed`.** Pasting a long command into a terminal inserts a newline at an arbitrary position, and once `sed` receives the truncated expression it reports:
>
> ```text
> sed: -e expression #1, char 85: unterminated `s' command
> ```
>
> When the command line is unavoidable, assemble the value from short fragments, each short enough not to wrap, and confirm with `echo` before applying:
>
> ```bash
> N='mgmt01=192.168.100.202'
> N="$N,gpu01=snorlax@192.168.100.215"
> N="$N,gpu02=yang@192.168.100.216"
> N="$N,gpu03=snorlax@192.168.100.217"
> echo "$N"
> sudo sed -i "s|^LELAB_CLUSTER_NODES=.*|LELAB_CLUSTER_NODES=$N|" /etc/robot-platform/lelab.env
> ```
>
> The `sed` script must use double quotes here, because `$N` has to expand.

The file is read by systemd as an `EnvironmentFile` and **only takes effect on a service restart**, so after editing it you must run:

```bash
grep -n LELAB_CLUSTER_NODES /etc/robot-platform/lelab.env
sudo systemctl restart lelab-platform
```

### LELAB_JOB_CACHE_ROOT

The directory `LELAB_JOB_CACHE_ROOT` points at must exist on **every worker** and be writable by `robot-train`:

```bash
sudo install -d -o robot-train -g robotdata -m 0750 /var/lib/robot-platform/cache
```

Slurm points `HOME` at the `robot-train` home directory, which does not exist on the workers, so jobs that cache into `~` (torch hub backbone weights, HF, wandb) fail on that node. The same local path on each node is enough; shared storage is not needed. When unset, it falls back to `HF_HOME`.

**This one produces no error at submission time and only fails once a job is scheduled onto a node where the directory is missing**, which makes it easy to miss when adding a node.

## 4. Generate the leLab SSH key

The following commands run on `mgmt01` only. If the file already exists, do not overwrite it — first check whether it is the key that is currently authorized.

```bash
sudo test -e /etc/robot-platform/lelab_ssh_key || \
  sudo ssh-keygen \
    -q \
    -t ed25519 \
    -N '' \
    -C lelab-gpu-probe \
    -f /etc/robot-platform/lelab_ssh_key

sudo chown \
  robot-train:robotdata \
  /etc/robot-platform/lelab_ssh_key \
  /etc/robot-platform/lelab_ssh_key.pub
sudo chmod 0600 /etc/robot-platform/lelab_ssh_key
sudo chmod 0644 /etc/robot-platform/lelab_ssh_key.pub
```

Install the public key into the login account of **every** worker (note the accounts may differ):

```bash
ssh-copy-id -f -i /etc/robot-platform/lelab_ssh_key.pub snorlax@192.168.100.215
ssh-copy-id -f -i /etc/robot-platform/lelab_ssh_key.pub yang@192.168.100.216
ssh-copy-id -f -i /etc/robot-platform/lelab_ssh_key.pub snorlax@192.168.100.217
```

Keeping `-f` here matters. By default `ssh-copy-id` may try to open the private key of the same name, and the private key is readable only by `robot-train`, so running it as an ordinary user gives `Permission denied`; `-f` installs exactly the public key given.

If `ssh-copy-id` is unavailable, paste it manually on the target node:

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
editor ~/.ssh/authorized_keys      # paste the single line from lelab_ssh_key.pub
chmod 600 ~/.ssh/authorized_keys
ssh-keygen -lf ~/.ssh/authorized_keys
```

The public key is one very long line. **If it wraps when pasted into the terminal, you get a key that can never match, with no error at all** — the symptom is a later `Permission denied (publickey)`. Use the last command to confirm the new key parses correctly.

If the currently logged-in user cannot traverse `/etc/robot-platform` to read the public key, copy it to a temporary file readable only by that user and delete it after installation:

```bash
sudo install -o "$USER" -g "$(id -gn)" -m 0600 \
  /etc/robot-platform/lelab_ssh_key.pub \
  /tmp/lelab_ssh_key.pub
ssh-copy-id -f -i /tmp/lelab_ssh_key.pub snorlax@192.168.100.215
rm -f /tmp/lelab_ssh_key.pub
```

## 5. Verify and install the worker host fingerprints

Authorizing the SSH public key and trusting the server host key are two different things. Even after `ssh-copy-id` succeeds, the systemd service can still fail to connect with `Host key verification failed`.

Two failure symptoms must be told apart; their fixes are completely different:

| Error | Cause | Section to read |
|---|---|---|
| `Permission denied (publickey,password)` | the host key passed, the public key is not in the remote `authorized_keys` | section 4 |
| `Host key verification failed` | nothing to do with the public key; `robot-train`'s known_hosts lacks that node | this section |

### 5.1 Scan each node from mgmt01

**Scan one node at a time into a separate file, and do not add `-H`.** `-H` hashes the hostname with a different salt every time, so rescanning the same machine produces several lines that look different and cannot be deduplicated; a hashed line also cannot be matched against the fingerprint seen on the node, which gives up manual verification entirely.

```bash
ssh-keyscan -T 5 -t ed25519 192.168.100.215 2>/dev/null > /tmp/kh215
ssh-keyscan -T 5 -t ed25519 192.168.100.216 2>/dev/null > /tmp/kh216
ssh-keyscan -T 5 -t ed25519 192.168.100.217 2>/dev/null > /tmp/kh217

ssh-keygen -lf /tmp/kh215
ssh-keygen -lf /tmp/kh216
ssh-keygen -lf /tmp/kh217
```

Each output looks like:

```text
256 SHA256:EWn7POFCGoIXDCnTAHRp4LQw1b68yiwLUj0HlpRII7Q 192.168.100.216 (ED25519)
```

The informational messages from `ssh-keyscan` (`# 192.168.100.216:22 SSH-2.0-OpenSSH_8.9p1 ...`) go to stderr and never reach the file; seeing them is normal.

### 5.2 Read the real fingerprint on each worker's console

```bash
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
```

```text
256 SHA256:EWn7POFCGoIXDCnTAHRp4LQw1b68yiwLUj0HlpRII7Q root@gpu02 (ED25519)
```

**Compare only the `SHA256:` field.** The trailing comment will always differ: the scan result labels the queried IP, while the node itself labels the key comment. The leading `256` and the `(ED25519)` should also match.

If the fingerprints differ, stop — whatever answers at that IP is not the machine you think it is. Do not skip the comparison.

### 5.3 Install known_hosts for robot-train

Prepare the directories the first time:

```bash
sudo install -d -o robot-train -g robotdata -m 0750 /home/robot-train
sudo install -d -o robot-train -g robotdata -m 0700 /home/robot-train/.ssh
```

Once verified, **append** (`-a`); do not overwrite entries for existing nodes:

```bash
sudo -u robot-train tee -a /home/robot-train/.ssh/known_hosts < /tmp/kh215 >/dev/null
sudo -u robot-train tee -a /home/robot-train/.ssh/known_hosts < /tmp/kh216 >/dev/null
sudo -u robot-train tee -a /home/robot-train/.ssh/known_hosts < /tmp/kh217 >/dev/null
rm -f /tmp/kh215 /tmp/kh216 /tmp/kh217
```

Use `sudo -u robot-train tee -a` here, **not** `sudo ... >>`: the redirection is performed by your current shell under your own identity rather than `robot-train`, which either fails to write or creates a file with the wrong owner that SSH then ignores outright.

Confirm each one was written:

```bash
for ip in 192.168.100.215 192.168.100.216 192.168.100.217; do
  sudo -u robot-train ssh-keygen -F "$ip" -f /home/robot-train/.ssh/known_hosts
done
```

All of these commands run on `mgmt01`, because the process initiating SSH is `robot-train` on `mgmt01`.

## 6. Verify SSH as the service account

Still on `mgmt01`, run this once against **every** worker. It matches exactly the connection the leLab service makes; your own user being able to SSH does not mean `robot-train` can:

```bash
for target in snorlax@192.168.100.215 yang@192.168.100.216 snorlax@192.168.100.217; do
  echo "[$target]"
  sudo -H -u robot-train ssh \
    -o BatchMode=yes \
    -i /etc/robot-platform/lelab_ssh_key \
    "$target" \
    nvidia-smi --query-gpu=name,memory.total,memory.free \
      --format=csv,noheader,nounits
done
```

Each must return GPU information without a password. `BatchMode=yes` forbids any interaction, so an unknown host key fails outright instead of prompting for confirmation — that is intentional. On failure, do not restart leLab first; check according to the error:

| Error | Cause |
|---|---|
| `Identity file ... not accessible` | the private key path, owner or permissions |
| `Permission denied (publickey,password)` | **the host key passed**; the public key is not in that node's `authorized_keys` (section 4) |
| `Host key verification failed` | nothing to do with the public key; `robot-train`'s known_hosts lacks that node or the fingerprint changed (section 5) |
| timeout | the IP, the SSH service or the firewall |

The first two are easy to confuse: as soon as you see `Permission denied`, the host key part is already correct and known_hosts needs no further attention.

## 7. Check the cluster API

```bash
sudo systemctl restart lelab-platform

curl --noproxy '*' -fsS http://127.0.0.1:8000/cluster/status | \
  jq -c '.nodes[] | {name,address,reachable,slurm_state,eligible,memory_free_mb,reason}'
curl --noproxy '*' -fsS \
  http://127.0.0.1:8000/cluster/templates | jq
```

Each node should look like:

```json
{"name":"gpu02","address":"yang@192.168.100.216","reachable":true,"slurm_state":"idle","eligible":true,"memory_free_mb":47752,"reason":null}
```

`/cluster/status` probes live on every request, so after editing `authorized_keys` or ending a GPU process there is **no need to restart leLab** — just request it again. Only changes to `/etc/robot-platform/lelab.env` need a restart.

What each node field means:

| Field | Meaning |
|---|---|
| `slurm_state` | the node state from `sinfo` |
| `reachable` | whether the local or SSH `nvidia-smi` succeeded |
| `compute_processes` | the current number of GPU compute processes |
| `eligible` | whether leLab is allowed to select this node |
| `reason` | the immediate reason it cannot be selected |

A node is `eligible: true` only when all of these hold:

- its Slurm state is `idle`;
- the GPU probe is reachable;
- there is no CUDA compute process;
- free VRAM satisfies the selected template.

If Slurm says `idle` but `compute_processes` is greater than 0, someone is usually running CUDA outside Slurm through an SSH or desktop session. **That node stays `eligible: false` and `auto` scheduling will never pick it.** To investigate, run on `mgmt01` (substituting the SSH target of the node in question):

```bash
sudo -H -u robot-train ssh \
  -o BatchMode=yes \
  -i /etc/robot-platform/lelab_ssh_key \
  snorlax@192.168.100.215 \
  nvidia-smi \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits
```

Then inspect the PID on that node:

```bash
ps -o user,pid,ppid,lstart,cmd -p <PID>
```

Confirm what the process is and who owns it before deciding whether its owner should stop it. Do not kill an unknown process. Once the process exits, leLab does not need restarting — just request `/cluster/status` again.

If the occupant is a remote desktop tool (RustDesk, TeamViewer, Sunlogin and similar), the right fix is not to close it but to add the process name to leLab's graphics process allow list — `graphics_patterns` in `_probe_node` in `apps/lelab/lelab/cluster.py`. `rustdesk` is already there, which is why `mgmt01` reports 0 even while running it.

## 8. Model templates

The active templates:

```text
/etc/robot-platform/model-templates.json
```

The templates constrain what a user can select:

- the LeRobot policy type;
- the Python training environment;
- the Slurm partition;
- the minimum free VRAM;
- the CPU and memory request.

In phase one, a template's `id` should equal its `policy_type`, so that no arbitrary command is exposed to the web. After editing:

```bash
sudo systemctl restart lelab-platform
curl --noproxy '*' -fsS \
  http://127.0.0.1:8000/cluster/templates | jq
```

## 9. NAS datasets and the jobs directory

leLab recognizes LeRobot datasets that contain `meta/info.json`:

```text
/mnt/robot_platform/datasets/team/pick-cube/
├── meta/info.json
├── data/
└── videos/
```

Check that the API found them:

```bash
curl --noproxy '*' -fsS \
  http://127.0.0.1:8000/datasets | jq
```

Every Slurm job writes:

```text
/mnt/robot_platform/jobs/<job-id>/
├── job.json
├── job.sbatch
├── log.jsonl
├── slurm.out
└── run/checkpoints/
```

`LELAB_OUTPUT_ROOT` must be visible at the same absolute path on **every** worker, otherwise that node cannot write logs and checkpoints. After adding a node, first confirm `findmnt /mnt/robot_platform` is healthy on it and that the QNAP allow list contains its IP.

## 10. The first training job

Before real use, pick a small dataset and the ACT template, set a very short number of training steps, and verify:

1. the UI lists the dataset;
2. at least one node is `eligible: true`;
3. after submission the job appears in `squeue`;
4. `slurm.out` and `log.jsonl` keep updating;
5. a checkpoint appears in the output directory;
6. Stop triggers `scancel`;
7. the UI can resume from a complete checkpoint.

If web submission fails, look at all of these together:

```bash
journalctl -u lelab-platform -f
squeue
scontrol show job <SlurmJobID>
```

## 11. Access and security boundary

The pilot address:

```text
http://192.168.100.202:8000
```

Port 8000 is currently not designed as a public-facing entry point. Before going live, add a reverse proxy, HTTPS, authentication and access control. The leLab private key may only be used for GPU probing; the commands it is allowed to run should later be restricted in `authorized_keys` on `gpu01`.
