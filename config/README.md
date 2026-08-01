# 配置文件与 Git 跟踪规则

本仓库统一采用以下规则：

- `*.example`：可复制的配置模板，必须由 Git 跟踪，不得包含真实密码、Token 或私钥。
- `*.template`：由脚本渲染的静态模板，必须由 Git 跟踪。
- 静态集群配置：对所有部署一致的文件由 Git 跟踪。
- 活动配置：包含站点地址、真实硬件参数或本地调整，只保留在部署主机并由 `.gitignore` 忽略。
- 生成配置：由脚本重新生成，不提交。
- 密钥与密码：不提交，也不通过 NAS 公共目录或聊天传递。

## 文件关系

| Git 跟踪的来源 | 本地或安装后的目标 | 目标是否跟踪 | 生成方式 |
|---|---|---:|---|
| `config/site.env.example` | `config/site.env` | 否 | 每台主机复制后核对；各 Slurm 节点必须使用相同的集群 UID/GID、节点名和地址 |
| `config/lelab.env.example` | `/etc/robot-platform/lelab.env` | 工作区外 | `15-install-lelab-platform.sh` 首次安装时复制，之后由管理员维护 |
| `config/slurm/nodes.conf.example` | `config/slurm/nodes.conf` | 否 | 填入各节点真实的 `slurmd -C` 输出 |
| `config/slurm/slurm.conf.template` | `config/slurm/slurm.conf.generated` | 否 | `render-slurm-config.sh` 根据 `site.env` 和 `nodes.conf` 生成 |
| `config/slurm/cgroup.conf` | `/etc/slurm/cgroup.conf` | 工作区外 | Controller/Worker 安装脚本复制 |
| `config/slurm/gres.conf` | `/etc/slurm/gres.conf` | 工作区外 | Controller/Worker 安装脚本复制 |
| `deploy/management/.env.example` | `deploy/management/.env` | 否 | `bootstrap.sh` 首次运行时生成随机数据库密码；example 仅用于说明字段 |
| `apps/lelab/config/model-templates.json.example` | `/etc/robot-platform/model-templates.json` | 工作区外 | leLab 安装脚本首次复制，之后由管理员维护 |
| `deploy/systemd/*.service.example` | `/etc/systemd/system/*.service` | 工作区外 | 对应角色安装脚本复制 |

## 初始化本地配置

```bash
cp config/site.env.example config/site.env
cp config/slurm/nodes.conf.example config/slurm/nodes.conf
```

`config/site.env` 被 Git 忽略，因此通过 `git clone` 部署其他节点时必须单独复制或重新创建。不要把管理机生成的 `deploy/management/.env`、Munge 密钥或 leLab SSH 私钥复制进本仓库。

生成 Slurm 配置：

```bash
./scripts/cluster/render-slurm-config.sh
```

生成的 `config/slurm/slurm.conf.generated` 也被 Git 忽略；请通过受控临时通道分发给 Worker，并在安装完成后删除临时副本。

## 活动配置不会自动覆盖

以下安装脚本采用“文件不存在时才创建”的策略：

- `15-install-lelab-platform.sh` 不覆盖 `/etc/robot-platform/lelab.env`；
- `15-install-lelab-platform.sh` 不覆盖 `/etc/robot-platform/model-templates.json`；
- `bootstrap.sh` 不覆盖 `deploy/management/.env`；
- `30-install-collector.sh` 不覆盖 `/etc/robot-platform/collector.env`。

因此修改仓库中的 `*.example` 后，已经安装的主机不会自动同步。应人工比较并编辑活动配置，再重启对应服务。

当前 leLab 节点映射中，左侧是 Slurm NodeName，右侧是 SSH 目标：

```bash
LELAB_CLUSTER_NODES=mgmt01=192.168.100.202,gpu01=snorlax@192.168.100.215
```

完整 SSH 设置见 [leLab 集群 Web](../docs/07-lelab-cluster-web.md)。
