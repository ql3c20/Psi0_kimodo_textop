> Publication update: the current release is `icra2027`. See repository `ICRA2027.md` for actual repositories, pinned commits, exclusions and validation. Older upload plans below are historical.

# Task3 GR00T → SONIC：GitHub 审计与跨服务器迁移流程

审计日期：2026-09-06。目标是上一份 Task3 LQB 评测命令的 `mujoco_external_isaac` 闭环，不包含 Kimodo / TextOp 主链，也不涉及实机。当前结论：**GitHub 上的现有分支尚不足以复现本机；需要先发布本地增量代码，再搬权重和场景数据。**

本次执行了 `git ls-remote`、`git fetch` 和逐文件 blob 比较；没有 commit/push，没有修改现有分支或暂存区。下面的发布命令是待执行流程。审计限定在实际配置的 GitHub 仓库/分支和明确标出的文件，不等于扫描全 GitHub 的所有历史分支。

## 1. 到底需要拉哪些仓库

| 组件 | GitHub | 新服务器需要范围 | 定位 |
|---|---|---|---|
| GR00T 推理 | https://github.com/ql3c20/Isaac-GR00T-rtc | **完整主体仓库**；模型包、processor、安装配置都要 | 当前基线 `rtc-rot6d59-support`；名称含 rot6d59，但也提供 SONIC 推理 |
| SIMPLE 评测 | https://github.com/ql3c20/SIMPLE_eval_env | **完整主体仓库 + 所需 submodules** | 建议放在 Psi0 的 `third_party/SIMPLE`，由父仓库锁定 commit |
| Psi0 调度与 HTTP server | https://github.com/ql3c20/Psi0_kimodo_textop | **只用 `scripts/deploy/` 等调度文件**；实际操作建议普通 clone，以保留 SIMPLE submodule 指针 | 不需要安装 Psi0 训练环境，不需要拉 Psi0 训练数据/权重 |
| Isaac viewer + MuJoCo 资产 | https://github.com/zhutengjie/HumanoidVLA_MJ | **子集**：`scripts/deploy/task3_isaac_hssd_viewer.py` + `mujoco/model/` 资源；可 sparse checkout 脚本，资产另外传 | 当前 viewer 未上传；无需完整项目环境 |
| SONIC release decoder | https://github.com/ql3c20/SONIC_my 或 NVlabs 上游 | **只需要本机 `gear_sonic_deploy/policy/release/model_decoder.onnx`** | 本次链路无需 clone/install 整个 GR00T-WholeBodyControl；decoder 不在已核对 fork/main 的 Git tree |
| gear_sonic | https://github.com/songlin/gear_sonic | SIMPLE 下的完整子模块包 | 锁定 `5b71b42ff6849cd121d93c58a7b508af8777c6c3` |
| decoupled_wbc | https://github.com/songlin/decoupled_wbc | SIMPLE 下的完整子模块包及资源 | 锁定 `83ac6281d0607727e0730b9926d2afea8abde4c3` |

另外，SIMPLE 的安装/导入还有 `unitree_sdk2_python`、`openpi-client`、XRoboToolkit、CuRobo 等依赖。按 `.gitmodules` 锁定版本初始化，按 SIMPLE README 安装。不要因为只跑仿真就随意删除依赖，当前包注册和导入并未完全按功能延迟加载。`third_party/evdev` 是 SIMPLE 已跟踪的普通源码目录，不需要额外创建仓库。

**不需要** Kimodo 仓库、Kimodo checkpoint/LLM2Vec、TextOp tracker/VAE 权重、实机 RoboJudo/OptiTrack/PICO 仓库。SIMPLE 中 `textop_tracker_adapter.py` 仍被 SONIC 导入用于关节映射等公共逻辑，必须带源码，但 SONIC 不实例化 TextOp VAE。

最低主体是 GR00T + SIMPLE；Psi0 提供调度脚本，HumanoidVLA_MJ 提供一个 viewer 和资源。为减少路径修改，下面仍保留原来的目录结构。

## 2. 当前 GitHub 上传状态

### 2.1 已在线确认的远端分支

| 仓库/分支 | GitHub 当前 commit | 能否直接复现本机 |
|---|---|---|
| Psi0 `simple_eval` | `f01a6994e43c57b3987c44ea6176a880abc4af72` | 不能：本地改动和新文件未包含 |
| SIMPLE `simple_eval` | `ee9a181e1268dc13fc8e0b99d1aa53416381756d` | 不能：Task3/外部 Isaac 等缺失 |
| Isaac-GR00T `rtc-rot6d59-support` | `b581c6d8aab42144ad4a61a3e7bc8064b286f366` | 不能代表当前代码：4 个推理相关文件有差异 |
| HumanoidVLA_MJ `feat/vla-kimodo-replay` | `afb35c108c5633704378ba67fd9e5bcdab5233a7` | viewer 和 trash 资产未在该树中 |
| SONIC_my `main` | `c08fc7bbc39efd710448dd58b663f049f18494de` | 该树没有本次 decoder 文件 |

`git push` 只推送 commit，不会带走工作区中 modified/untracked 文件。即使显示 Everything up-to-date，也不能说明本机链路已上传。

### 2.2 逐文件证据

下表相对路径以“组件”对应仓库为根。`缺失` 表示当前核对远端树没有此路径；`不同` 表示本地文件 blob 与远端不同。

| 组件 | 文件 | 比较结果 |
|---|---|---|
| Psi0_kimodo_textop | `scripts/deploy/eval_lqb_simple_task3_sonic.sh` | 缺失 |
| Psi0_kimodo_textop | `scripts/deploy/fullstate_task1_gr00t_n17_sonic_eval.sh` | 不同 |
| Psi0_kimodo_textop | `scripts/deploy/fullstate_task3_gr00t_rot6d59_kimodo_textop_eval.sh` | 不同 |
| Psi0_kimodo_textop | `scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh` | 不同 |
| Psi0_kimodo_textop | `scripts/deploy/gr00t_n17_sonic_server.py` | 不同 |
| Psi0_kimodo_textop | `scripts/deploy/gr00t_n17_prefix_rtc.py` | 一致 |
| Psi0_kimodo_textop | `scripts/deploy/task3_lqb_sonic_textop_full_pipeline.md` | 缺失 |
| Psi0_kimodo_textop | `.gitmodules` | 一致 |
| SIMPLE | `scripts/validate_fullstate_recordings.py` | 缺失 |
| SIMPLE | `src/simple/tasks/g1_fullstate_20260804_task3.py` | 缺失 |
| SIMPLE | `src/simple/tasks/g1_fullstate_recordings.py` | 缺失 |
| SIMPLE | `src/simple/tasks/g1_fullstate_20260615_task1.py` | 不同 |
| SIMPLE | `src/simple/tasks/__init__.py` | 不同 |
| SIMPLE | `src/simple/baselines/gr00t_n17_sonic.py` | 不同 |
| SIMPLE | `src/simple/baselines/textop_tracker_adapter.py` | 不同 |
| SIMPLE | `src/simple/cli/eval_decoupled_wbc.py` | 不同 |
| SIMPLE | `src/simple/engines/external_isaac_ego.py` | 缺失 |
| SIMPLE | `src/simple/engines/mujoco.py` | 不同 |
| SIMPLE | `src/simple/envs/base_dual_env.py` | 不同 |
| SIMPLE | `src/simple/envs/sonic_loco_manip.py` | 不同 |
| SIMPLE | `src/simple/envs/__init__.py` | 不同 |
| SIMPLE | `src/simple/envs/video_writer.py` | 不同 |
| SIMPLE | `src/simple/envs/wrappers/video_recorder.py` | 不同 |
| SIMPLE | `src/simple/evals/tui.py` | 不同 |
| SIMPLE | `pyproject.toml` | 一致 |
| Isaac-GR00T-rtc | `gr00t/policy/gr00t_policy.py` | 不同 |
| Isaac-GR00T-rtc | `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` | 不同 |
| Isaac-GR00T-rtc | `gr00t/deployment/modes.py` | 不同 |
| Isaac-GR00T-rtc | `gr00t/utils/video_utils.py` | 不同 |
| Isaac-GR00T-rtc | `pyproject.toml` | 一致 |
| HumanoidVLA_MJ | `scripts/deploy/task3_isaac_hssd_viewer.py` | 缺失 |
| HumanoidVLA_MJ | `mujoco/model/task_assets/2real_asset/trash/step_trash_can.xml` | 缺失 |
| GR00T-WholeBodyControl | `gear_sonic_deploy/policy/release/model_decoder.onnx` | 缺失 |

完整 blob 证据保存在同目录 `task3_sonic_migration/github_file_audit.json`。

### 2.3 子模块不是缺源码，但必须正确拉取

本机 `SIMPLE/third_party/gear_sonic` 和 `decoupled_wbc` 内没有 `.git`；在其目录执行 `git rev-parse HEAD` 会向上找到 SIMPLE，不能拿这个结果当子模块版本。上一份文档已更正。

本次从 GitHub 获取上述 **SIMPLE 锁定 commit**，对 `.py/.toml/.yaml/.yml/.json/.sh` 文件做比较：gear_sonic 57 个、decoupled_wbc 254 个全部相同，无缺失、无额外 Python 文件。比较未覆盖所有二进制资产及任意后缀文件，因此权重仍应独立校验。证据为 `task3_sonic_migration/submodule_source_audit.json`。

它们的远端最新 HEAD 已变化；用父仓库 gitlink 锁定值，不要 `submodule update --remote`。本次也确认 unitree_sdk2_python、openpi-client、XRoboToolkit 的 HTTPS 远端可访问；其他递归模块的全新服务器安装仍需实际验证。

## 3. 权重与数据：另外同步

GitHub 保存源码、安装说明和清单；大权重、recording、USD/mesh 从当前服务器通过 rsync/SCP 或已有对象存储分发。这里只核对 Git tree，没有证据说明这些文件已作为 GitHub Release 或其他云端附件发布，所以不能给一个未经验证的权重下载链接。

以 `/home/ubuntu/yzh` 为源根，需同步：

| 相对路径 | 用途 / 范围 |
|---|---|
| `ckpt/gr00tn17/task3_isaac_sonic_lqb/checkpoint-60000/` | 完整推理 checkpoint；可排除 `global_step60000/` optimizer、RNG、scheduler 等续训文件，但最稳妥先保留完整目录 |
| `Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B/` | backbone、tokenizer、processor；解引用 symlink 或连 blobs 一起传 |
| `GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx` | 在线 SONIC decoder，仅这一文件 |
| `Psi0_kimodo_textop/third_party/SIMPLE/third_party/decoupled_wbc/sim2mujoco/resources/robots/g1/policy/` | Balance/Walk ONNX，构造 stabilizer 时仍会加载，即使 SKIP_STABILIZE=1 |
| `mujoco_recordings/20260804_task3_lqb/` | 100 条 CSV 与 model_snapshot；初次迁移保留完整目录以免断掉资源引用 |
| `Psi0_kimodo_textop/third_party/SIMPLE/data/robots/g1_sonic/` | MuJoCo 机器人 XML 和 mesh；当前不在 SIMPLE 普通 Git 文件中 |
| `Psi0_kimodo_textop/third_party/SIMPLE/data/scenes/hssd/` | 包含 102344250_local.usd 的背景资源；保守同步整个 hssd，之后可按 USD 依赖闭包缩减 |
| `HumanoidVLA_MJ_backup/HumanoidVLA_MJ/mujoco/model/` | 任务/模型资源目录；保守同步整个 model，避免 recording XML 外部 include / mesh 丢失；Task3 直接物理资产为 task_assets/2real_asset/trash |
| `HumanoidVLA_MJ_backup/HumanoidVLA_MJ/output/task3_lqb_isaac_lerobot/scene3/task3_lqb_all_groot_sonic_release_train/meta/lighting.jsonl` | 只需要该灯光清单，在线不需要整份训练视频/Parquet |

上述完整目录范围是保守迁移集合，不声称已解析全部 USD/MJCF 外部引用。若资源包含目录外绝对引用，在新服务器需要修正副本或保留对应路径；不能全局替换原始文件。环境 `.venv` 不直接复制，目标机器重新建立。当前 Isaac `VERSION` 是 `6.0.1-rc.7+release.42383.32955d8d.gl`，此 viewer 应优先使用兼容安装，不能假定任意 Isaac 版本 API 一致。

## 4. 源服务器：准备代码发布分支

建议所有相关仓库发布 `deploy/task3-sonic-20260906`。下面在**新的隔离目录**做准备，不动正在使用的工作树和已有暂存区。此流程创建的是源码增量快照，发布前需看 diff。

四份上传清单已经生成在 `scripts/deploy/task3_sonic_migration/`：

- `psi-files.txt`：入口/公共 wrapper/SONIC server/文档。
- `simple-files.txt`：当前 SIMPLE 的本地源码增量及预检脚本。包括注册模块直接 import 的其他 Task 文件，避免只上传 Task3 导致注册时 ImportError。
- `groot-files.txt`：4 个本地推理核心改动，不包括训练产物和部署 wheels。
- `mj-files.txt`：Isaac viewer 单文件。

这些是本次快照清单，若源文件继续改动应重新检查清单与 diff。下列步骤从仓库当前 HEAD 创建新 clone，再显式覆盖清单文件。

```bash
set -euo pipefail
export SRC=/home/ubuntu/yzh
export PUB=/home/ubuntu/yzh/task3_sonic_publish_20260906
export REL=deploy/task3-sonic-20260906
export MAN="$SRC/Psi0_kimodo_textop/scripts/deploy/task3_sonic_migration"
test ! -e "$PUB"  # 新目录；已经存在时先检查，不覆盖旧发布目录
mkdir -p "$PUB"

GIT_LFS_SKIP_SMUDGE=1 git clone --no-hardlinks "$SRC/Psi0_kimodo_textop" "$PUB/Psi0_kimodo_textop"
GIT_LFS_SKIP_SMUDGE=1 git clone --no-hardlinks "$SRC/Psi0_kimodo_textop/third_party/SIMPLE" "$PUB/SIMPLE_eval_env"
GIT_LFS_SKIP_SMUDGE=1 git clone --no-hardlinks "$SRC/Isaac-GR00T-rtc" "$PUB/Isaac-GR00T-rtc"
# MJ 仓库只 checkout 所需脚本，避免展开其大资产目录
GIT_LFS_SKIP_SMUDGE=1 git clone --no-hardlinks --no-checkout "$SRC/HumanoidVLA_MJ_backup/HumanoidVLA_MJ" "$PUB/HumanoidVLA_MJ"
git -C "$PUB/HumanoidVLA_MJ" sparse-checkout init --cone
git -C "$PUB/HumanoidVLA_MJ" sparse-checkout set scripts/deploy

for repo in Psi0_kimodo_textop SIMPLE_eval_env Isaac-GR00T-rtc HumanoidVLA_MJ; do
  git -C "$PUB/$repo" switch -c "$REL"
done

git -C "$PUB/Psi0_kimodo_textop" remote set-url origin https://github.com/ql3c20/Psi0_kimodo_textop.git
git -C "$PUB/SIMPLE_eval_env" remote set-url origin https://github.com/ql3c20/SIMPLE_eval_env.git
git -C "$PUB/Isaac-GR00T-rtc" remote set-url origin https://github.com/ql3c20/Isaac-GR00T-rtc.git

rsync -a --files-from="$MAN/psi-files.txt" "$SRC/Psi0_kimodo_textop/" "$PUB/Psi0_kimodo_textop/"
rsync -a --files-from="$MAN/simple-files.txt" "$SRC/Psi0_kimodo_textop/third_party/SIMPLE/" "$PUB/SIMPLE_eval_env/"
rsync -a --files-from="$MAN/groot-files.txt" "$SRC/Isaac-GR00T-rtc/" "$PUB/Isaac-GR00T-rtc/"
rsync -a --files-from="$MAN/mj-files.txt" "$SRC/HumanoidVLA_MJ_backup/HumanoidVLA_MJ/" "$PUB/HumanoidVLA_MJ/"
rsync -a "$MAN/" "$PUB/Psi0_kimodo_textop/scripts/deploy/task3_sonic_migration/"
```

HumanoidVLA_MJ 的 origin 属于 `zhutengjie`，本次只确认读取权限，未测试写权限。如果你已有该仓库写权限，可用该 origin 推新分支。否则先在 GitHub 将该仓库 fork 到自己的账号（网页 Fork；或已登录的 GitHub CLI 执行下列命令），再设置发布 clone 的 origin：

```bash
gh repo fork zhutengjie/HumanoidVLA_MJ --clone=false
# 这里按 ql3c20 是目标账号举例；若 fork 到另一账号，替换它
git -C "$PUB/HumanoidVLA_MJ" remote set-url origin https://github.com/ql3c20/HumanoidVLA_MJ.git
```

这个 fork 是待创建/确认的发布目标，本次未验证其是否已存在。也可将 viewer 放入自有部署仓库保持 `scripts/deploy/` 相对布局，但需要相应调整下载命令。

## 5. 检查、提交和上传（子仓库先于父仓库）

先在发布 clone 中只暂存明确清单。shell 语法检查只检查真正可执行的 wrapper，不能对带 Markdown 围栏的命令备忘录当作普通脚本验证。

```bash
git -C "$PUB/SIMPLE_eval_env" add --pathspec-from-file="$MAN/simple-files.txt"
git -C "$PUB/Isaac-GR00T-rtc" add --pathspec-from-file="$MAN/groot-files.txt"
git -C "$PUB/HumanoidVLA_MJ" add --pathspec-from-file="$MAN/mj-files.txt"
git -C "$PUB/Psi0_kimodo_textop" add --pathspec-from-file="$MAN/psi-files.txt"
git -C "$PUB/Psi0_kimodo_textop" add scripts/deploy/task3_sonic_migration

for repo in SIMPLE_eval_env Isaac-GR00T-rtc HumanoidVLA_MJ Psi0_kimodo_textop; do
  git -C "$PUB/$repo" diff --cached --check
  git -C "$PUB/$repo" diff --cached --stat
  git -C "$PUB/$repo" diff --cached --name-status
done
bash -n "$PUB/Psi0_kimodo_textop/scripts/deploy/fullstate_task1_gr00t_n17_sonic_eval.sh"
bash -n "$PUB/Psi0_kimodo_textop/scripts/deploy/fullstate_task3_gr00t_rot6d59_kimodo_textop_eval.sh"
bash -n "$PUB/Psi0_kimodo_textop/scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh"
```

看过 `git diff --cached` 后提交、推送前三个仓库：

```bash
for repo in SIMPLE_eval_env Isaac-GR00T-rtc HumanoidVLA_MJ; do
  git -C "$PUB/$repo" commit -m "Add current Task3 SONIC evaluation runtime"
  git -C "$PUB/$repo" push -u origin "$REL"
  git -C "$PUB/$repo" ls-remote origin "refs/heads/$REL"
done
```

再更新 Psi0 的 SIMPLE gitlink，保证别人 clone 父仓库时拿到刚上传的新 SIMPLE commit：

```bash
SIMPLE_SHA=$(git -C "$PUB/SIMPLE_eval_env" rev-parse HEAD)
git -C "$PUB/Psi0_kimodo_textop" update-index --cacheinfo "160000,$SIMPLE_SHA,third_party/SIMPLE"
git -C "$PUB/Psi0_kimodo_textop" config -f .gitmodules submodule.third_party/SIMPLE.branch "$REL"
git -C "$PUB/Psi0_kimodo_textop" add .gitmodules
git -C "$PUB/Psi0_kimodo_textop" diff --cached --submodule=short
```

将三个子发布 commit 记录到父仓库，并最终推送：

```bash
python3 - <<'PYCODE'
import json, os, subprocess
from pathlib import Path
pub=Path(os.environ['PUB'])
lock={r:subprocess.check_output(['git','-C',str(pub/r),'rev-parse','HEAD'],text=True).strip()
      for r in ['SIMPLE_eval_env','Isaac-GR00T-rtc','HumanoidVLA_MJ']}
p=pub/'Psi0_kimodo_textop/scripts/deploy/task3_sonic_migration/release_commits.json'
p.write_text(json.dumps(lock,indent=2)+'\n')
PYCODE
git -C "$PUB/Psi0_kimodo_textop" add scripts/deploy/task3_sonic_migration/release_commits.json
git -C "$PUB/Psi0_kimodo_textop" diff --cached --check
git -C "$PUB/Psi0_kimodo_textop" commit -m "Publish Task3 SONIC launchers and pin evaluation runtime"
git -C "$PUB/Psi0_kimodo_textop" push -u origin "$REL"
git -C "$PUB/Psi0_kimodo_textop" ls-remote origin "refs/heads/$REL"
```

逐个确认 `ls-remote` 的 SHA 等于发布 clone 的 HEAD。不要用 force push；如果远端同名分支已存在且不兼容，重新取独立发布分支名。原始工作区有预先暂存的其他修改，本流程不会把它们顺带提交。

## 6. 新服务器：拉代码

下面 `$BASE` 可以改成有权限的磁盘路径；先创建空目标目录。发布流程完成后才有这个 REL 分支。

```bash
set -euo pipefail
export BASE=/home/ubuntu/yzh
export REL=deploy/task3-sonic-20260906
mkdir -p "$BASE"
cd "$BASE"
GIT_LFS_SKIP_SMUDGE=1 git clone -b "$REL" https://github.com/ql3c20/Psi0_kimodo_textop.git
GIT_LFS_SKIP_SMUDGE=1 git -C "$BASE/Psi0_kimodo_textop"   -c url.https://github.com/.insteadOf=git@github.com:   submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 git clone -b "$REL" https://github.com/ql3c20/Isaac-GR00T-rtc.git

mkdir -p "$BASE/HumanoidVLA_MJ_backup"
GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --sparse -b "$REL"   https://github.com/ql3c20/HumanoidVLA_MJ.git   "$BASE/HumanoidVLA_MJ_backup/HumanoidVLA_MJ"
git -C "$BASE/HumanoidVLA_MJ_backup/HumanoidVLA_MJ" sparse-checkout set scripts/deploy
```

如 HumanoidVLA_MJ 实际发在原仓库或其他 fork，应替换上面的 URL。SIMPLE 已通过 Psi0 submodule 下载，不再单独 clone 到另一个路径。HTTPS rewrite 只作用于这条命令，不改全局 Git 配置。不要使用 `--remote`，它会偏离锁定的子模块 commit。

按父仓库保存的 lock 固定 GR00T 和 viewer，验证 SIMPLE gitlink：

```bash
python3 - <<'PYCODE'
import json, os, subprocess
from pathlib import Path
base=Path(os.environ['BASE'])
lock=json.loads((base/'Psi0_kimodo_textop/scripts/deploy/task3_sonic_migration/release_commits.json').read_text())
for name,folder in [('Isaac-GR00T-rtc','Isaac-GR00T-rtc'),('HumanoidVLA_MJ','HumanoidVLA_MJ_backup/HumanoidVLA_MJ')]:
    subprocess.run(['git','-C',str(base/folder),'checkout','--detach',lock[name]],check=True)
p=base/'Psi0_kimodo_textop/third_party/SIMPLE'
actual=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
assert actual==lock['SIMPLE_eval_env'], (actual,lock)
print('Release commits verified')
PYCODE
```

## 7. 新服务器：同步权重、数据与环境

先从源服务器同步本节清单中各目录。以下示例在**源服务器执行**，修改账号/IP/目标根；`-L` 把 symlink 的实际内容传过去，`-R` 保留源根以下路径；不使用 `--delete`。

```bash
export TARGET=ubuntu@NEW_SERVER
export TARGET_BASE=/home/ubuntu/yzh
cd /home/ubuntu/yzh
ssh "$TARGET" "mkdir -p '$TARGET_BASE'"
rsync -aLR --partial --info=progress2   ./ckpt/gr00tn17/task3_isaac_sonic_lqb/checkpoint-60000   ./Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B   ./GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx   ./Psi0_kimodo_textop/third_party/SIMPLE/third_party/decoupled_wbc/sim2mujoco/resources/robots/g1/policy   ./mujoco_recordings/20260804_task3_lqb   ./Psi0_kimodo_textop/third_party/SIMPLE/data/robots/g1_sonic   ./Psi0_kimodo_textop/third_party/SIMPLE/data/scenes/hssd   ./HumanoidVLA_MJ_backup/HumanoidVLA_MJ/mujoco/model   ./HumanoidVLA_MJ_backup/HumanoidVLA_MJ/output/task3_lqb_isaac_lerobot/scene3/task3_lqb_all_groot_sonic_release_train/meta/lighting.jsonl   "$TARGET:$TARGET_BASE/"
```

源/目标身份与路径设为简单可信值，不在 shell 参数里填带引号或命令替换的文本。传完用相同 rsync 参数加 `--dry-run --checksum` 检查是否还会传文件；这一步读取所有大文件，可能较慢。对象存储中转也保留同样目录层级和 checksum。

新服务器安装同架构的 Python 3.10、uv、Git LFS、NVIDIA 驱动/CUDA 运行依赖，以及兼容 Isaac Sim。按本地 README 的基线命令：

```bash
cd "$BASE/Isaac-GR00T-rtc"
uv sync --python 3.10
cd "$BASE/Psi0_kimodo_textop/third_party/SIMPLE"
UV_HTTP_TIMEOUT=3000 GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups --index-strategy unsafe-best-match
bash scripts/install_curobo.sh
```

这是仓库安装流程，不是已经在你的另一台服务器通过的环境验收。GPU/架构/驱动未知，flash-attn、torchcodec、ORT CUDA 的兼容性需在目标机器确认；新 ARM 服务器不可直接套用 x86 wheel。当前 `.venv` 来自本机，不建议复制整个环境。若使用 Git LFS 获取资产，需要对实际拥有该文件的仓库执行相应 `git lfs pull`；本流程已把关键本地 ONNX 单独同步，不依赖缺失的 LFS 下载链接。

## 8. 新服务器：1 集 smoke test

各终端先设置公共根（若 BASE 不同，**原命令块中其余硬编码绝对路径也必须改成 BASE**）：

```bash
export BASE=/home/ubuntu/yzh
export PSI0_ROOT="$BASE/Psi0_kimodo_textop"
export SIMPLE_ROOT="$PSI0_ROOT/third_party/SIMPLE"
export GR00T_ROOT="$BASE/Isaac-GR00T-rtc"
export HUMANOID_VLA_MJ_ROOT="$BASE/HumanoidVLA_MJ_backup/HumanoidVLA_MJ"
export ISAAC_PYTHON=/home/ubuntu/isaacsim/python.sh
cd "$PSI0_ROOT"
```

先预检（此脚本本机 100 条已通过；新机器必须再跑）：

```bash
cd "$SIMPLE_ROOT"
PYTHONPATH=src .venv/bin/python scripts/validate_fullstate_recordings.py   --task 3 --recordings-dir "$BASE/mujoco_recordings/20260804_task3_lqb"
.venv/bin/python - <<'PYCODE'
import onnxruntime as ort
import simple.tasks
import gear_sonic, decoupled_wbc
print('providers:',ort.get_available_providers())
print('gear_sonic:',gear_sonic.__file__)
print('decoupled_wbc:',decoupled_wbc.__file__)
PYCODE
```

按 `scripts/deploy/eval_lqb_simple_task3_sonic.sh` 的 3 个独立代码块依次启动 GR00T、Isaac、eval。首次把 eval 的 `RUN_TAG` 换成新 smoke 名、`NUM_EPISODES=1`、`TASK3_RECORDING_INDEX=0`；保留 40/40、prefix=0、相机/灯光/初始姿态、MAX_EPISODE_STEPS=500 等原参数。CPU 不支持 16-31 时用有效 `PROCESS_CPUSET`/`ISAAC_CPUSET`/`PREFLIGHT_CPUSET` 覆盖。

验收依次看：GR00T checkpoint 正确且 prediction_horizon=40 / prefix_rtc=false；Isaac 等待 MuJoCo 和同步帧；eval 实际 action `(40,78)`、帧持续推进；产出新 RUN_TAG 下的 `results.jsonl`、`summary.json`、视频；无 policy_or_isaac_error。1 集未成功不直接说明部署失败，应区分控制结果与服务/数据错误。链路 smoke 通过后再回到 seed42 的100集设置。

SONIC server 的 `/config` 可能仍显示默认 execution_horizon=34：wrapper 只在 prefix 开启时传该 CLI 参数。本次 prefix 关闭时实际执行40由 eval 环境 `GR00T_EXECUTION_HORIZON=40` 决定。完整解释见 `task3_lqb_sonic_textop_full_pipeline.md`。

## 9. 本次交付与未执行项

已完成：真实远端读取、主要文件 blob 差异表、两组 pinned submodule 源码比较、上传清单、可照做的发布/下拉/迁移流程、上一份文档子模块版本纠正。

未执行：创建 fork、发布分支 commit/push、搬运大权重、另一台服务器安装与评测。这里给出的是具体操作流程，不把它们描述为已经上传完成。
