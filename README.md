# LLM for Medication Recommendation

## 项目结构
- data
    - '/data/finetune_data.ipynb': 根据负样本比例生成完整的微调训练集和验证集
    - '/data/finetune_data_ips.ipynb': 根据负样本比例生成完整的微调训练集和验证集，使用加权负采样方法，减少popularity bias
    - '/data/finetune_data_fix_drug.ipynb': 根据负样本比例和**药物名称列表**生成微调训练集和验证集
- evaluation
    - '/evaluation/1_overall_evaluation_vllm.py': 使用VLLM加速的评估，加载一个模型和完整测试集，评测总体性能
    - '/evaluation/2_each_drug_evaluation_vllm.py': 使用VLLM加速的评估，加载一个模型和一个药物列表，评估每个药物的性能
    - '/evaluation/3_each_lora_evaluation_vllm.py': 使用VLLM加速的评估，加载多个模型，每个模型负责一个药物，评估每个药物的性能
    - '/evaluation/call_evaluation.ipynb': 调用上述三个评估函数
    - '/evaluation/case_study.ipynb': 统计微调数据的情况
    - '/evaluation/Evaluate.ipynb': 不用VLLM加速的评估
- log：评测结果
- Models：存放基础模型权重
    - llama-2-7b
        - config.json: 需要把"max_position_embeddings"设置为4096，否则会报错
- output：存放微调模型权重及训练日志
    - each_drug_v1: 不使用history，只使用disease procedure short title训练的模型，使用了最初版本的prompt template
    - each_drug_history_short_title: 使用history和disease procedure short title训练的模型
    - each_drug_history_concise_title1: 使用history和disease procedure concise title训练的模型.
    - each_drug_history_concise_2: 使用history和disease procedure concise title训练的模型，title进行了更新。
    - each_drug_long_title: 使用history和disease procedure long title训练的模型，效果一般
    - overall_history_concise_1: 使用所有药物的数据，使用history和disease procedure concise title训练的模型，以为加入了history可以改善popularity bias，但是效果不好。
    - para_search_each_drug_50_53：搜索batch_size和learning_rate的结果，以为增大batch_size可以使训练更稳定，但是效果不好。batch_size减小的效果也不好，训练会更不稳定。
- src
    - '/src/finetune.py': 用完整药物列表构成的微调数据集微调模型
    - '/src/finetune_fix_drug.py': 用指定药物列表构成的微调数据集微调模型
    - '/src/run.sh': 运行finetune.py和finetune_fix_drug.py的脚本
    - '/src/job.sh': 运行finetune.py和finetune_fix_drug.py的脚本，用于提交到超算集群服务器
- utils
    - '/utils/templates': 用于生成prompt的模板，训练和测试都要用到
        - 'utils/templates/llama-2-7b_v1.json': 最初版本的prompt template, 只使用disease procedure short title
        - 'utils/templates/llama-2-7b_v2.json': 使用history和disease procedure short title
    - '/utils/evaluate_utils.py': 评估的工具函数
    - '/utils/file_utils.py': 文件读写的工具函数
    - '/utils/merge_lora_utils.py': 用于合并lora的工具函数
    - '/utils/process_utils.py': 用于处理数据的工具函数
    - '/utils/prompt_utils.py': 用于生成prompt的工具函数
    
## 开发计划
- [x] 实现逐药物的评测
- [x] 完善了logger，可以看到trainer的日志了。
- [x] 实现逐药物的微调
- [x] 基于每个药物独立的lora，实现逐药物的评测
- [ ] VLLM多卡推理 tensor_parallel_size


## 实验进展与计划
- [x] 给每个药物独立训练一个lora
- [ ] 用每个药物的数据逐个训练
- [ ] shuffle ips数据做全量训练


## 常用git命令：

### git config
- git config --global user.name "Your Name"
- git config --global user.email "Your Email"
- git config --global core.editor vim
- git config --list     # 查看配置信息

### 仓库初始化
- git clone git@...    # 克隆仓库
- git remote -v        # 查看远程仓库信息

### 仓库操作
- git status            # 查看仓库状态
- git add <file>        # 添加文件到暂存区
- git commit -m "..."   # 提交到本地仓库
- git commit -am "..."  # 添加到暂存区并提交到本地仓库
- git commit --amend    # 修改最后一次提交
- git push <branch>     # 推送到远程仓库
- git pull <branch>     # 从远程仓库拉取
- git log               # 查看提交历史

- git branch            # 查看分支
- git branch <name>     # 创建分支
- git checkout <name>   # 切换分支
- git checkout -b <name> # 创建并切换分支
- git merge <name>      # 合并某分支到当前分支
- git merge --squash <name> # 把待合并分支的更新合并成一个merge到现有分支上
- git branch -d <name>  # 删除分支
- git branch -D <name>  # 强行删除分支

- git stash             # 保存工作现场
- git stash list        # 查看保存的工作现场
- git stash apply       # 恢复工作现场
- git stash drop        # 删除工作现场
- git stash pop         # 恢复并删除工作现场


