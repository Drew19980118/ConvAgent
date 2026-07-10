# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em, qa_em_format_conv
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em_format_conv.compute_score_em
    elif data_source in ['topiocqa', 'qrecc']:
        return qa_em_format_conv.compute_score_f1
    elif data_source in ['wow', 'cast2020', 'cast2022', 'ikat2023', 'ikat2024', 'mtfiqa', 'mtclapnq', 'mtgovt', 'mtibmcloud']:
        return qa_em_format_conv.compute_score_f1
    elif data_source in ['slupart/qrecc-rewrite-mistral', 'slupart/topiocqa-rewrite-mistral', 'slupart/cast20-rewrite-mistral', 'slupart/cast22-rewrite-mistral', 'slupart/ikat23-rewrite-mistral']:
        return qa_em_format_conv.compute_score_f1
    elif data_source in ['ultrachat', 'inscit']:
        # return qa_em_format_conv.compute_score_f1
        return qa_em_format_conv.f1_score_ConvAgent
    elif data_source in ['faithdial', 'coqa', 'md2d', 'mantis', 'icconv']:
        return qa_em_format_conv.compute_score_f1
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, structure_format_score=0., final_format_score=0., retrieval_score=0., format_score=0., rewrite_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.structure_format_score = structure_format_score
        self.final_format_score = final_format_score
        self.retrieval_score = retrieval_score
        self.rewrite_score = rewrite_score

    # For ConvAgent
    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        full_texts = data.meta_info.get('full_texts')
        if full_texts is None:
            # 安全 fallback（不含 information，但避免崩溃）
            print("Warning: 'full_texts' not in meta_info. Constructing from prompts+responses.")
            prompts = data.batch['prompts']
            responses = data.batch['responses']
            full_texts = []
            for i in range(len(data)):
                full_ids = torch.cat([prompts[i], responses[i]])
                full_text = self.tokenizer.decode(full_ids, skip_special_tokens=False)
                full_texts.append(full_text)

        for i in range(len(data)):
            data_item = data[i]
            full_text = full_texts[i]
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            # ---------- 1. 解析模型输出，确定动作类型并截断文本 ----------
            # 检测 clarify
            clarify_match = re.search(r'<clarify>.*?</clarify>', full_text, re.DOTALL)
            has_clarify = clarify_match is not None

            # 提取所有 answer（取最后一个）
            answer_matches = list(re.finditer(r'<answer>(.*?)</answer>', full_text, re.DOTALL))
            pred_answer = None
            last_answer_match = None
            if answer_matches:
                last_answer_match = answer_matches[-1]
                pred_answer = last_answer_match.group(1).strip()

            # 初始化变量
            truncated_text = full_text  # 默认不截断
            action_type = 'none'  # 可选: 'clarify', 'noanswer', 'answer', 'none'
            outcome_score = 0.0

            if has_clarify:
                # 优先级最高：clarify
                action_type = 'clarify'
                # 截断到 </clarify> 结束
                end_pos = clarify_match.end()
                truncated_text = full_text[:end_pos]
                outcome_score = 0.0
            elif pred_answer is not None:
                # 有 answer，判断是否为 noanswer
                noanswer_keywords = ['sorry', 'not find', 'no information', 'unable to find', 'did not find']
                has_noanswer = any(kw in pred_answer.lower() for kw in noanswer_keywords)
                if has_noanswer:
                    action_type = 'nonanswer'
                else:
                    action_type = 'answer'
                # 截断到最后一个 </answer> 结束
                end_pos = last_answer_match.end()
                truncated_text = full_text[:end_pos]
                # outcome 仅当正常 answer 时稍后计算，否则保持 0
            else:
                # 无任何终止标签
                action_type = 'none'
                # 不截断，outcome 为 0

            # ---------- 2. 提取信息增益文本（基于截断后的文本） ----------
            info_matches = re.findall(r'<information>(.*?)</information>', truncated_text, re.DOTALL)
            info_text = ' '.join([m.strip() for m in info_matches])

            # ---------- 3. 计算 R_outcome（仅当 action_type == 'answer'） ----------
            if action_type == 'answer' and pred_answer:
                for gt in ground_truth:
                    score = compute_score_fn(pred_answer, gt['response'])
                    if score > outcome_score:
                        outcome_score = score
            # 其他情况 outcome_score 已为 0

            # ---------- 4. 计算 R_IG（只要有信息） ----------
            ig_score = 0.0
            if info_text:
                for gt in ground_truth:
                    score = compute_score_fn(info_text, gt['response'])
                    if score > ig_score:
                        ig_score = score

            # ---------- 5. 计算 R_MIA ----------
            mia_score = 0.0
            # 确定模型预测的动作
            if has_clarify:
                pred_action = 'clarify'
            elif action_type == 'noanswer':
                pred_action = 'noanswer'
            elif action_type == 'answer':
                pred_action = 'answer'
            else:
                pred_action = None  # 无任何终止标签

            # 从 ground_truth 中提取所有真实动作（去重）
            true_actions = list(set([gt.get('action', 'answer') for gt in ground_truth]))

            if pred_action is not None:
                if pred_action in true_actions:
                    mia_score = 1.0
                else:
                    mia_score = -0.5
            else:
                # 若模型未输出任何标签，不给予 MIA 奖惩（可根据需要设为 -0.5 或 0）
                mia_score = 0.0

            # ---------- 6. 加权求和 ----------
            total_reward = outcome_score + 0.5 * (ig_score + mia_score)

            # ---------- 7. 赋值给最后一个有效 token ----------
            prompt_len = data_item.batch['prompts'].shape[0]
            valid_len = data_item.batch['attention_mask'][prompt_len:].sum().item()
            if valid_len > 0:
                reward_tensor[i, valid_len - 1] = total_reward

        return reward_tensor

    # For ChatR1
    # def __call__(self, data: DataProto):
    #     """We will expand this function gradually based on the available datasets"""
    #
    #     # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
    #     if 'rm_scores' in data.batch.keys():
    #         return data.batch['rm_scores']
    #
    #     reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
    #
    #     # all_scores = []
    #
    #     already_print_data_sources = {}
    #
    #     for i in range(len(data)):
    #         data_item = data[i]  # DataProtoItem
    #
    #         prompt_ids = data_item.batch['prompts']
    #
    #         prompt_length = prompt_ids.shape[-1]
    #
    #         valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
    #         valid_prompt_ids = prompt_ids[-valid_prompt_length:]
    #
    #         response_ids = data_item.batch['responses']
    #         valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
    #         valid_response_ids = response_ids[:valid_response_length]
    #
    #         # decode
    #         sequences = torch.cat((valid_prompt_ids, valid_response_ids))
    #         sequences_str = self.tokenizer.decode(sequences)
    #
    #         ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
    #
    #         # select rm_score
    #         data_source = data_item.non_tensor_batch['data_source']
    #         compute_score_fn = _select_rm_score_fn(data_source)
    #
    #         score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth,
    #                                  structure_format_score=self.structure_format_score,
    #                                  final_format_score=self.final_format_score,
    #                                  retrieval_score=self.retrieval_score,
    #                                  format_score=self.format_score,
    #                                  rewrite_score=self.rewrite_score)
    #
    #         reward_tensor[i, valid_response_length - 1] = score
    #         # all_scores.append(score)
    #
    #         if data_source not in already_print_data_sources:
    #             already_print_data_sources[data_source] = 0
    #
    #         if already_print_data_sources[data_source] < self.num_examine:
    #             already_print_data_sources[data_source] += 1
    #             print(sequences_str)
    #
    #     return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    # ray.get(main_task.remote(config))
    main_task(config)


# @ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # import pydevd_pycharmn
    # pydevd_pycharm.settrace('localhost', port=34769, stdoutToServer=True, stderrToServer=True)

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, 
                              structure_format_score=config.reward_model.structure_format_score, 
                              final_format_score=config.reward_model.final_format_score,
                              retrieval_score=config.reward_model.retrieval_score,
                              rewrite_score=config.reward_model.rewrite_score)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
