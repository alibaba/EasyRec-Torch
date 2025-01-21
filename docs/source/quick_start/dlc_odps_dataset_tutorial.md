# DLC Tutorial (MaxCompute Table）

此文档提供在阿里云的PAI-DLC（分布式训练容器）中使用TorchEasyRec读取MaxCompute（MC）表进行模型训练的步骤和配置示例。

## 环境准备

### 创建数据集

进入[PAI控制台](https://pai.console.aliyun.com/)，点击 **AI资源管理-数据集** -> **创建数据集**。选择数据存储为**阿里云文件存储(NAS)**，挂载到`/mnt/data`下。任务运行时，会从挂载路径下读取训练数据和配置文件以及写入模型检查点等。

### 加载训练数据到MaxCompute

1. [安装并配置MaxCompute客户端](https://help.aliyun.com/zh/maxcompute/user-guide/maxcompute-client?spm=a2c4g.11186623.0.0.7e4e4be52z9TXQ#section-vd2-4me-7uu)

1. 获取并执行脚本来创建数据表并上传数据至MaxCompute

```bash
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/upload_data.sh
bash upload_data.sh ${ODPS_PROJECT_NAME}
```

### 前置条件

- 在[MaxCompute控制台](https://maxcompute.console.aliyun.com/)的「租户管理」->「租户属性」页面打开**开放存储(Storage API)开关**。
- 「租户管理」->「新增成员」给相应用户授予「admin」权限；或参考[租户权限](https://help.aliyun.com/zh/maxcompute/user-guide/overview-1#cabfa502c288o)文档，精细授予用户Quota的使用权限

## 创建DLC任务

#### 配置文件

配置文件以DIN为例 [multi_tower_din_taobao_dlc_mc.config](https://tzrec.oss-cn-beijing.aliyuncs.com/config/quick_start/multi_tower_din_taobao_dlc_mc.config)

### 配置任务

进入[PAI控制台](https://pai.console.aliyun.com)，并选择需要使用的工作空间，点击 **模型开发与训练-分布式训练(DLC)**，点击创建任务。

**节点镜像** 选择官方镜像`torcheasyrec:0.6.0-pytorch2.5.0-gpu-py311-cu121-ubuntu22.04`

**数据集配置** 选择刚新建的NAS数据集

**资源配置** 选择框架为PyTorch，任务资源我们以选择单机8卡V100为例（建议优先选择单机多卡机型，需要多机多卡训练时建议选择带RDMA的机型）

**角色信息** 选择**PAI默认角色**

**执行命令** 如下

### 训练命令

设置`ODPS_ENDPOINT`为对应的MaxCompute服务地址，执行训练命令：

```bash
cd /mnt/data
wget https://tzrec.oss-cn-beijing.aliyuncs.com/config/quick_start/multi_tower_din_taobao_dlc_mc.config
# 安装tzrec并启动训练
pip install tzrec==${TZREC_NIGHTLY_VERSION} -f http://tzrec.oss-cn-beijing.aliyuncs.com/release/nightly/repo.html --trusted-host tzrec.oss-cn-beijing.aliyuncs.com
ODPS_ENDPOINT=http://service.{region}.maxcompute.aliyun-inc.com/api \
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
--nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
-m tzrec.train_eval \
--pipeline_config_path /mnt/data/multi_tower_din_taobao_dlc_mc.config \
--train_input_path odps://{project_name}/tables/taobao_data_train \
--eval_input_path odps://{project_name}/tables/taobao_data_test \
--model_dir /mnt/data/multi_tower_din_odps_dlc
```

- `--pipeline_config_path`: 训练用的配置文件路径。
- `--train_input_path`: 指定训练用的MaxCompute的表。
- `--eval_input_path`: 指定评估用的MaxCompute表。

MaxCompute表按如下格式设置：
odps://{project}/tables/{table_name}/{partition}，多表按逗号分隔
如果单表需要设置多个分区，可以用&简写，来分隔多个分区，odps://{project}/tables/{table_name}/{partition1}&{partition2}

### 评估命令

完成模型训练后，进行模型评估：

```bash
ODPS_ENDPOINT=http://service.{region}.maxcompute.aliyun-inc.com/api \
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
--nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
-m tzrec.eval \
--pipeline_config_path /mnt/data/multi_tower_din_odps_dlc/pipeline.config \
--eval_input_path odps://{project_name}/tables/taobao_data_test
```

- --pipeline_config_path: 评估用的配置文件
- --checkpoint_path: 指定要评估的checkpoint, 默认评估model_dir下面最新的checkpoint
- --eval_input_path: 指定评估用的MaxCompute表

### 导出模型

导出训练好的模型：

```bash
ODPS_ENDPOINT=http://service.{region}.maxcompute.aliyun-inc.com/api \
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
--nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
-m tzrec.export \
--pipeline_config_path /mnt/data/multi_tower_din_odps_dlc/pipeline.config \
--export_dir /mnt/data/multi_tower_din_odps_dlc/exported_model
```

- --pipeline_config_path: 导出用的配置文件
- --checkpoint_path: 指定要导出的checkpoint, 默认导出model_dir下面最新的checkpoint
- --export_dir: 导出到的模型目录

### 预测

使用导出的模型进行预测：

```bash
ODPS_ENDPOINT=http://service.{region}.maxcompute.aliyun-inc.com/api \
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
--nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
-m tzrec.predict \
--scripted_model_path /mnt/data/multi_tower_din_odps_dlc/exported_model \
--predict_input_path odps://{project_name}/tables/taobao_data_test \
--predict_output_path odps://{project_name}/tables/taobao_data_test_output \
--reserved_columns user_id,adgroup_id,clk
```

- --scripted_model_path: 要预测的导出模型
- --predict_input_path: 指定预测用的MaxCompute表
- --predict_output_path: 预测结果的输出MaxCompute表
- --reserved_columns: 预测结果中要保留的输入列
- --output_columns: 预测结果中的模型输出列

## 注意事项

1. 确保在MaxCompute控制台中已为相关用户授予必要的权限以访问数据表。
1. 在训练命令中，`ODPS_ENDPOINT`环境变量需设置为对应的MaxCompute服务地址。
