/data/zilu/QseekLLM/src/post_train/Compute_Cot用于构建第一阶段的算数表达式推理数据,我们需要保证数据覆盖广,推理指导够强,能够教会模型去计算一些简单的问题,另外我们需要保证训练与测试的题目是不含完全匹配的,也就是,不要让训练和测试存在重合

各种缓存放置于/data/zilu目录下,包括hf,uv...

uv环境在本目录下管理,不要泄露到其他目录

benchmark的详细说明参考/data/zilu/QseekLLM/src/post_train/docs/benchmark中的文档

训练数据的详细说明参考/data/zilu/QseekLLM/src/post_train/docs/traindata的文档

tokenizer使用/data/zilu/QseekLLM/src/llmtrain/qseek_digitsplit_base,而不是/data/zilu/QseekLLM/src/llmtrain/stage0_data_tokenizer_hf,我们切分数字去做

(目前有的东西存在了其他目录,你可以先看看内容了解一下再说)
预训练模型权重说明:
目前存储在
/data/zilu/fastrl/checkpoints/latest_infer_cpt
/data/zilu/fastrl/checkpoints/latest_infer_cpt_hf_vllm

/data/zilu/fastrl/checkpoints/latest_infer_ctx8k
/data/zilu/fastrl/checkpoints/latest_infer_ctx8k_hf_vllm

/data/zilu/fastrl/checkpoints/latest_infer_ctx16k
/data/zilu/fastrl/checkpoints/latest_infer_ctx16k_hf_vllm

/data/zilu/fastrl/checkpoints/latest_infer_pretrained
/data/zilu/fastrl/checkpoints/latest_infer_pretrained_hf_vllm

目前的数据集下载脚本/data/zilu/fastrl/scripts/download_benchmark.py,/data/zilu/fastrl/scripts/download_train.py


整体计划:
1.对base模型执行初步的SFT,这部分SFT旨在强化模型的意图理解,指令跟随,逻辑思维能力,语言表达能力,以及初步对齐模型的对话格式
- 本阶段需要执行过程跟踪,本阶段过程跟踪,不跟踪数值指标,只对一些训练样本外的问答问题做一些测试,保存模型输出结果,这样方便我们跟踪模型的性能情况
- 另外训练在A800上执行,推理评测在A4000上执行,这样评测不阻塞训练过程
- 不过注意A800和A4000的实际标号,这个你需要测试一下,nvidia-smi可能不准确

2.第二阶段,我们借助Compute_Cot构建各种基本数学的SFT数据,利用这些数据强化模型的符号推演,算数运算能力,并将think包裹引入其中,将推演过程包装为Cot,另外本阶段还要混入一些常规语言任务的SFT数据,防止语言退化现象出现
- 本阶段,我们首先要构建数据,可以先看看 Compute_Cot提供了什么
- SFT阶段采用动态课程学习的方法,即一开始各个领域内容平衡配比,然后每隔若干step后,进入过程中评测,这个也是异步拉起的,不要阻塞训练,然后当这个评测评测完后,会给训练侧一个反馈,训练侧抽取数据的时候,就会调节采样配比
- 另外记得混杂一些常规通用SFT,保证语言不过度退化
- 过程中评测测试集的计算准确度,自然语言对话测试,小规模数学习题测试(看看我们这种光使用合成Cot数据带来的"计算能力",是否可以泛化到问题求解上)

3.在给定的数学习题数据上做多阶段课程SFT,大致思路是
"第一阶段:简单的多,难的少"
"第二阶段逐渐增加难题占比"
...
- 过程中异步拉起benchmark评测,同样用A4000卡,跟踪策略熵,回答准确率Pass@k=1,8,记录完整的模型输入,输出,方便给我追溯观测

4.最终目标,经过前面一系列努力后,我们的模型具备了一定的数学基础推理能力,我们借Verl实现高效异步GRPO训练
- 采用Lora微调
- 先通过同步版本做初步测试,确定超参数选取
- 同步版本,训练放在A800上,采样放在A4000上,采样可以开多卡,然后我们需要在同步版本中,跟踪采样的时间开销,训练的时间开销,权重同步的时间开销
- 同步版本smoke后,我们需要搞清楚几件事情,时间开销,GRPO是否产生了效果,超参数选取
- 然后,我们基于各个阶段的时间开销,选取合适的异步GRPO参数配置,训练若干轮次初步验证可行性
- 然后,做系统的效率瓶颈分析:基于同步/异步版本的实测数据,把每步耗时拆解归因(采样生成,训练forward/backward,权重导出与传输,sleep/wake显存切换,prefix cache清理,调度空泡等),写成瓶颈分析文档,明确各部分占比与扩展性(模型变大/卡数变多时谁先恶化)
- 最后,根据瓶颈分析结论选取优化目标并尝试实现,候选方向包括:只同步Lora块而不传输完整权重(verl已内置TensorLoRARequest热挂载机制,peft_config+base_sync_done路径,优先验证现成路径是否够用),提高异步重叠率,削减sleep/wake开销等;若分析表明某方向收益甚微,记录结论并放弃,不为优化而优化
- 已有的硬件实测参考:本机8卡无NVLink,全PCIe Gen3 x16,GPU间P2P实测约5GB/s(组内组间无差别);1.7B bf16全量权重3.4GB,估算4卡A4000全量同步约2-5秒/次,LoRA-only约0.1-0.3秒/次


一些开发的规范需求:
要有日志文档记录,方便工作交接,这样别人通过日志文档就能搞清楚你干了什么,在哪改的,改了什么,现在执行到哪一步骤了
对于关键的接口,应该直接在文档中暴露,比如启动采样,启动训练,之类的东西,这样我们可以很快的搞清楚外部结构,再比如训练进度和状态查询接口,这个就很方便
总之,为自己搭建一个好的脚手架,也为人类搭建一个好的脚手架,这样当你额度达到的时候,我也可以手动接手任务
先不着急开发,把数据格式之类的弄清楚后,再开始开发,另外过程探索记录都可以写到/data/zilu/QseekLLM/src/post_train/docs里面
文件结构组织要简介优雅,便于检索,复查
