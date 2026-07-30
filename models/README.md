# Model artifacts

该目录只保存模型版本说明和本地开发占位文件，不提交模型权重。

正式模型应存放在外部模型仓库或对象存储中，并记录：

- 模型版本和校验值
- nnU-Net配置与训练折
- 训练数据清单版本
- 评估报告
- 发布时间和审核人
- 适用范围与已知限制

本机已接入的`Dataset002_BRATS19`权重说明见`docs/BRATS19_MODEL.md`。实际权重
位于被Git忽略的`runtime/nnunet/nnUNet_results`，不复制到本目录。
