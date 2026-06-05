# 开发过程
如果用采样的200GB数据来训练Tokenizer，数据大小太大，训练tokenizer的时候会爆内存，所以选择了50GB的规模，最终效果依旧不错
测试结果：/mnt/paper2any/ziyi/llmTrain/runs/stage0_data_tokenizer_hf/tokenizer_inspection.json

700M训练1.2B左右时，发现模型输出一直重复，于是调小了学习率衰减（min_lr_rario），并适当减小global_batch_size，让模型更新更频繁，重新训练v2版本

测评部分：由于只进行了预训练，因此我们尽量选择续写类的任务进行测评，以适配预训练的任务模式。选择Chinese WPLC（中文）与LAMBADA（英文）
结果：Latest下：Chinese WPLC：5%（人类准确率57.3%），LAMBADA：25%（GPT2 762M 准确率60.1%）
1B数据：Chinese WPLC 3%， LAMBADA 2.5%

我随机抽取了ChineseWebText2.0和CCI3-HQ的各100条数据以查看数据特点，发现ChineseWebText2.0的语料噪声更多，且低质量数据较多，因此我下载了chinese-fine-web-edu-v2.1中评分为3-5分的较高质量数据，用来替代大部分的ChineseWebText2.0数据，以缓解模型在中文输出上的偏网络营销、广告风格。另外，提高了中文wiki的数据占比，用以提供更多人类撰写的，且专业知识准确的知识。

1.7B模型训练到30B时，出现了明显的退化现象，模型输出结果不如以前自然了，但我并没有合适的评估模型能力的方法，因此我先从数据集中抽取了一小部分（每个源取6M数据）搭建了一个66M的验证集，用于检测模型的loss和PPL值，看模型性能是否有在提高。对5B-30B的checkpoint测试后，发现模型在验证集上的loss从5B-30B单调升高，而训练集loss在下降，明显出现了过拟合问题。我初步怀疑这是因为学习率衰减过慢导致的，因为我的学习率衰减是按照总共训练100B Tokens而设定的线性衰减速度，而模型仅在训练前期需要大学习率，后续应该快速衰减到更小的值以防止过参数化。于是我在300M的小模型上运行了一轮6B数据的测试，这轮把学习率的衰减范围改为6B，因此学习率的衰减会变快，我计划在这个实验配置下测试一下模型的val loss能否正常下降，以判断究竟是学习率的问题还是数据质量的问题。

在单纯的余弦衰减下，模型的loss能够单调下降，但在后期下降放缓，且这种学习率调度策略的灵活度有限，模型几乎只能在最初学习率较高的时候大量学习，而后续学习相当于只是微调，因此我想尝试更灵活的学习率调度策略。下一步测试WSD策略，首先warmup，然后在stable（固定学习率）下完成大部分Token的学习，最后再接入衰减阶段，让模型顺利收敛。

由于中文的数据质量一般，为了提高模型输出中文文本的质量，我又基于Chinese-Cosmopedia进行了一个小型的继续预训练（CPT），并保留40%的比重用于上阶段训练数据的replay，防止模型出现灾难性遗忘。一共使用1B数据。

接下来，使用较长的语料（书籍、论文以及长代码文件）进行上下文加长的训练，让模型见一些长文本，以帮助其上下文窗口扩展至16K。采用分阶段的扩展方式，首先用2B数据从4K扩展至8K，接下来用3B数据从8K扩展至16K。数据配比如下：长文本72%，其他防遗忘文本28%，长文本中又分为STEM风格（45%）与一般风格（27%），分别为数学、代码、论文等面向后续推理型后训练的数据，以及一般的书籍、长网页等长文本。


# 常用命令
训练命令：
python run.py train \
    --config configs/train/stage1_general_wsd_50b.yaml \
    --gpus all \
    --device cuda

双机 16 卡训练命令：

MCCL/NCCL环境变量、节点数、master地址默认从 `.env` 自动读取。由于 `.env` 在共享盘上，两台机器不要在里面写死 `LLMTRAIN_NODE_RANK`，启动时分别传 `--node-rank 0/1`。

节点0（master，以 `.env` 里的 `LLMTRAIN_MASTER_ADDR` 为准）：
```bash
python run.py train \
    --config configs/train/stage1_general_wsd_50b.yaml \
    --gpus all \
    --device cuda \
    --node-rank 0
```

节点1：
```bash
python run.py train \
    --config configs/train/stage1_general_wsd_50b.yaml \
    --gpus all \
    --device cuda \
    --node-rank 1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python run.py train \
    --config configs/train/stage1_general_wsd_50b.yaml \
    --gpus all \
    --nproc-per-node 7 \
    --device cuda \
    --node-rank 1 \
    --micro-batch-size 1 \
    --global-batch-size 504
```

注意：`master_addr` 使用节点0的 10.201.x.x 地址，不要使用 172.17.x.x；重启开发机后 10.201 地址可能变化，需要同步更新 `.env` 里的 `LLMTRAIN_MASTER_ADDR`。跨节点 NCCL/MCCL smoke test 已验证 8 卡/节点 all_reduce 正常，sum=16。

推理命令：
```bash
python run.py infer \
    --config configs/train/stage_ext_16k_1700m.yaml \
    --checkpoint /mnt/paper2any/ziyi/llmTrain/runs/stage_ext_16k/checkpoints/milestone_003000000000 \
    --device cuda \
    --dtype bf16 \
    --prompt "爱因斯坦是" \
    --temperature 0.4 \
    --max-new-tokens 256 \
    --stream \
    --kv-cache
```

导出checkpoint：
```bash
PYTHONPATH=src python -m torch.distributed.run --standalone --nproc-per-node 8 \
    tools/export_checkpoint.py \
    --config configs/train/stage1_general_resume_30b_wsd_50b.yaml \
    --checkpoint runs/stage1_general/checkpoints/latest \
    --output runs/stage1_general/checkpoints/latest_infer
```

测评：
```bash
python run.py eval \
    --config configs/train/stage_ext_16k_1700m.yaml \
    --checkpoint /mnt/paper2any/ziyi/llmTrain/runs/stage1_general_700m_v2/checkpoints/latest \
    --output-dir runs/eval_700m_v2 \
    --run-name stage1_700m_15B \
    --gpus all \
    --batch-size 4
```

# 问题发现

中文数据污染很严重：北京小吃生成如下
网 北京小吃网 是为京城内餐馆提供美食信息的网站.提供北京小吃信息.北京小吃网 为京城内餐馆提供美食信息的网站.北京小吃网 ,北京小吃网 是一家全国性的餐饮信息门户网站,是北京市政府重点扶持的餐饮企业.网站由北京小吃网 提供,北京小吃网 网站信息免费提供给用户,免费提供北京小吃信息,提供北京小吃信息.网站简介 : 北京小吃网 是以介绍、介绍北京小吃为主,以北京小吃信息为主,提供北京小吃信息,提供北京小吃信息,为京城内餐馆提供美食信息的网站..

火锅：
行业:火锅行业受资本青睐随着火锅行业的快速发展,火锅行业也迎来了发展的黄金时期,很多火锅行业创业者都纷纷开始创业。火锅行业作为火锅行业的细分行业,其行业发展前景也是非常广阔的。那么,火锅行业前景如何呢?投资火锅行业有哪些好的项目?1、投资火锅行业的好项目,是投资火锅行业的绝佳项目。投资火锅行业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资创业,投资

后续可以再引入一些高质量的中文数据，比如chinese-fine-web-edu-v2.1，另外，可以增加百科数据的占比，以强化常识知识，另外百科也属于高质量文本

用fineweb里面4-5以及3-4分的数据替换ChineseWebText2.0，以提高高质量文本比例



1.7B模型训练速度太慢，按课程作业截止的话做不完训练，因此考虑通过算子优化进行加速/显存节约
尝试了 LigerFusedLinearCrossEntropyLoss、LigerRMSNorm、LigerSwiGLUMLP，大幅节约了显存，因此可以关闭activationc_checkpoint，提高了50%的训练速度，每秒Tokens数量41K->62K。随后接入 FlashAttention varlen/GQA 路径：packed 数据不再构造巨大的 [B,S,S] mask，GQA 也不再复制 K/V，所以attention更省显存也更快。每秒Tokens数量62K->90K

继续优化（2026-05-25）：
- trainer 内移除 micro-step 每步的 D2H sync（loss.cpu()）与 all-reduce(consumed_tokens)，合并到 optimizer step 一次；FSDP grad_accum 开 no_sync。
- FSDP 加上 forward_prefetch / backward_prefetch=pre / use_orig_params=true（顺手修了 build_optimizer 按 name 拆 weight_decay 分组在 use_orig_params=false 下失效的隐藏 bug）。
- A/B 实测（每组跑 35 step，warm step 15-35 median tok/s，1.7B / 8×MetaX C500 64GB / mbs=4 / seq=4096）：
  - baseline (full_shard + compile + fused AdamW): 92.9K tok/s, peak mem 42.7GB
  - compile_model=false: 91.3K tok/s（沐曦上 inductor 几乎没收益）
  - AdamW foreach (fused=false): 92.9K tok/s（两者无差）
  - **shard_grad_op (ZeRO-2): 104K tok/s, peak mem 45.6GB** ✅
- 选 shard_grad_op，每秒Tokens数量90K->104K。
- 从 full_shard checkpoint 切到 shard_grad_op resume 验证通过：loss 连续、step 衔接、DCP 加载正常。

速度90K->100K

# TODO
