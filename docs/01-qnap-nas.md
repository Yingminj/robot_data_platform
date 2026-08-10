# QNAP NAS setup

**English** | [简体中文](01-qnap-nas.zh-CN.md)

The NAS is the authoritative file storage for raw H5 files, dataset versions, annotation exports, MLflow artifacts and model release bundles. It does not run PostgreSQL, the MLflow server or Slurm.

## 1. Storage layout

Preferably create a dedicated shared folder `robot_platform` on the QNAP, matching `NAS_EXPORT=/robot_platform` in `config/site.env.example`. If for now you have to keep using `/kmd_data_file`, create a dedicated subdirectory and change `NAS_EXPORT` to the NFS export path the QNAP actually publishes:

```text
/kmd_data_file/robot-platform/
├── incoming/
├── raw/
├── quarantine/
├── annotations/
├── datasets/
├── jobs/
├── mlflow-artifacts/
├── model-releases/
├── backups/
└── trash/
```

Do not move other projects out of an existing share root or change their permissions. The platform directories should stay isolated from existing data.

## 2. Phase-one NFS service

Enable NFSv4 in QTS and add host access rules for the shared folder:

| IP | Role | Pilot permission |
|---|---|---|
| `192.168.100.202` | mgmt01 (management + GPU) | read/write |
| `192.168.100.215` | gpu01 | read/write |
| `192.168.100.216` | gpu02 | read/write |
| `192.168.100.217` | gpu03 | read/write |

All machines use the same export and the same mount point, `/mnt/robot_platform`.

> **When adding a worker, its static IP must be added to this allow list first.** This is the easiest step to miss. The symptom is that `findmnt /mnt/robot_platform` prints nothing on the new node or the mount is read-only, while job submission raises no error — the failure only appears once a job is scheduled onto that node.

At minimum these must be created:

```text
datasets/
jobs/
mlflow-artifacts/
```

The pilot phase does not require the QNAP to build ACLs from Linux numeric UIDs/GIDs, and does not use member-level permissions. This deployment keeps the QNAP default of "map all users to guest" (all_squash): every platform account (`robot-ingest`, `robot-train`, in-container processes) is evaluated on the NAS as `guest`, and the goal is simply that the platform services on all nodes can read `datasets` and write their own `jobs/<job-id>`. It is still recommended to open access only to the allow-listed platform nodes rather than to the whole subnet.

In all_squash mode, confirm the following:

1. QTS → Control Panel → Privilege → Shared Folders → `robot_platform`: grant the `guest` account **RW** (guest is often denied by default; when it is denied, reads and writes fail for every platform account, with a symptom unrelated to Linux-side permissions).
2. The skeleton directories only need to be writable by guest. Since all client users are mapped to guest, guest ownership of the directories is enough for every platform service; they can be created directly from any mount point:

   ```bash
   sudo mkdir -p /mnt/robot_platform/{incoming,raw,quarantine,annotations,datasets,jobs,mlflow-artifacts,model-releases,backups,trash}
   ```

   If the directories already exist but guest cannot access them, prefer the QTS shared-folder permission UI to grant guest access to the platform share and these dedicated subdirectories. The client's root is also mapped to guest and normally cannot change server-side ownership from the mount. Do not run a recursive `chmod 0777` over an entire existing share.

3. In this mode every file on the NAS is owned by guest and there is no per-user audit trail; numeric UID/GID ACLs and setgid conventions are deferred to the data governance phase.

Note: Slurm itself still requires the training account to have the same UID/GID on all workers. `TRAIN_UID` and `DATA_GID` in `config/site.env` only settle the Slurm runtime identity and play no part in the NAS permission design at this stage.

## 3. Data protection

Configure at least:

- scheduled snapshots of the platform directories;
- a snapshot retention policy;
- tiered capacity alerts at 70%, 80% and 90%;
- disk, RAID and fan health alerts;
- an additional retention policy for the PostgreSQL backup directory;
- a second copy on a second storage device or on offline media.

NAS snapshots can recover an accidental deletion, but they are not a substitute for an independent backup.

## 4. NAS acceptance

On the management node, confirm:

```bash
showmount -e 192.168.100.184
findmnt /mnt/robot_platform
df -hT /mnt/robot_platform
```

Confirm on **every** host that the mount is `rw`. Verify with the platform training account:

```bash
sudo -u robot-train test -r /mnt/robot_platform/datasets
sudo -u robot-train test -w /mnt/robot_platform/jobs
```

QNAP UID/GID mapping, fine-grained ACLs and raw data protection are handled in the later formal data governance phase and are not prerequisites for bringing the phase-one training platform online.
