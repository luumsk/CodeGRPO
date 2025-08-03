# Paper: https://arxiv.org/pdf/2402.03300

import copy, random
import torch


def clone_model(model): 
    cloned = copy.deepcopy(model)
    return cloned

def sample_batch(dataset, batch_size=4):
    return random.sample(dataset, batch_size)

def sample_outputs(policy_model, question, group_size):
    outputs = []
    output_lengths = []

    for _ in range(group_size):
        output = policy_model.run(question)
        outputs.append(output)
        tokenized = policy_model.tokenizer.encode(
            output, 
            add_special_tokens=False, 
        )
        output_lengths.append(len(tokenized))
    return outputs, output_lengths

def get_groundtruth(question):
    return "ground_truth_answer_for_" + question

def reward_function(question, output):
    ground_truth = get_groundtruth(question)
    if output == ground_truth:
        return 1.0
    else:
        return 0.0

def compute_advantages(rewards, output_lengths):
    r = torch.tensor(rewards, dtype=torch.float32)
    mean = r.mean()
    std = r.std(unbiased=False)
    normalized_r = (r - mean) / (std + 1e-8)

    advantages = []
    for norm_r_i, length in zip(normalized_r, output_lengths):
        advantages.extend([norm_r_i] * length)
    return advantages


def update_policy(): pass



def grpo(
    pi_theta_init, # initial policy model
    r_phi, # reward function
    D, # dataset of questions
    epsilon, # clip parameter
    beta, # KL divergence coefficient
    mu, # number of GRPO optimization steps per batch
    I, # number of iterations
    M, # number of batches per iteration
    G # number of sample outputs per question     
):
    pi_theta = clone_model(pi_theta_init)
    
    for i in range(I):
        pi_ref = clone_model(pi_theta)

        for m in range(M):
            D_b = sample_batch(D)
            pi_theta_old = clone_model(pi_theta)

            all_outputs = []
            all_rewards = []
            all_advantages = []

            for q in D_b:
                outputs, output_lengths = sample_outputs(pi_theta_old, q, G)
                rewards = [r_phi(q, o) for o in outputs]
                advantages = compute_advantages(rewards, output_lengths)

                all_outputs.append(outputs)
                all_rewards.append(rewards)
                all_advantages.append(advantages)

            for _ in range(mu):
                pi_theta = update_policy()
    
    return pi_theta 