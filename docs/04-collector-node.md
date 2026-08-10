# Collector node installation

**English** | [简体中文](04-collector-node.zh-CN.md)

A collector node runs ROS2 collection, H5 conversion, local validation and the upload agent. By default it does not mount the NAS and uploads through the management node API.

## 1. Install the base components

```bash
cp config/site.env.example config/site.env
editor config/site.env
./scripts/00-audit-host.sh collector
sudo ./scripts/30-install-collector.sh --apply
```

The script creates:

```text
/var/spool/robot-data/
├── recording/
├── ready-to-upload/
├── uploading/
├── uploaded/
└── failed/
```

and:

```text
/etc/robot-platform/collector.env
/etc/systemd/system/robot-upload-agent.service
```

## 2. Components the project still has to provide

This workspace contains no collection application source, so the following still have to be installed:

- ROS2 plus the robot/camera drivers;
- the H5 converter;
- the H5 schema validator;
- the manifest generator;
- the upload agent;
- the node upload token issued by the platform.

The reserved installation path for the upload agent:

```text
/opt/robot-platform/bin/robot-upload-agent
```

The token path:

```text
/etc/robot-platform/upload.token
```

Recommended token permissions:

```bash
sudo chown root:robotdata /etc/robot-platform/upload.token
sudo chmod 0640 /etc/robot-platform/upload.token
```

Once the application is ready:

```bash
sudo systemctl enable --now robot-upload-agent
```

## 3. Collection state rules

- an H5 file being written may only live in `recording`;
- the validator runs only after the H5 file has been closed cleanly;
- the file moves to `ready-to-upload` only after the validator, the manifest and the SHA-256 have all succeeded;
- uploads use `.partial` files and resumable transfer;
- local files must not be cleaned up before the server returns `COMMITTED`;
- when local space drops below `COLLECTOR_MIN_FREE_GB`, stop new collection rather than deleting uncommitted data.

## 4. Network and permissions

A collector node only needs HTTPS 443 access to the management node. It does not need write permission on the NAS `raw` directory, and it does not need PostgreSQL, Redis, the MLflow server, Munge or Slurm.

If the collected data contains faces, voices or sensitive views of the workspace, anonymization and approval must happen before it enters an external data warehouse; local raw data still follows the organization's retention rules.
