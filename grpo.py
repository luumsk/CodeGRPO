# Paper: https://arxiv.org/pdf/2402.03300

import copy, random
import torch


def copy_model(model): 
    return copy.deepcopy(model)

def sample_batch(dataset, batch_size=4):
    return random.sample(dataset, batch_size)

def generate_outputs(policy_model, question, group_size):
    outputs = []
    output_lengths = []
    output_tokenized = []

    for _ in range(group_size):
        output = policy_model.generate(question)
        outputs.append(output)
        tokenized = policy_model.tokenizer.encode(
            output, 
            add_special_tokens=False, 
        )
        output_lengths.append(len(tokenized))
        output_tokenized.append(tokenized)
    return outputs, output_lengths, output_tokenized

def get_groundtruth(question):
    return "ground_truth_answer_for_" + question


# Outcome supervision (for the entire output)
def reward_outcome(question, output):
    ground_truth = get_groundtruth(question)
    if output == ground_truth:
        return 1.0
    else:
        return 0.0

def adv_outcome(rewards, output_lengths):
    r = torch.tensor(rewards, dtype=torch.float32)
    mean = r.mean()
    std = r.std(unbiased=False)
    normalized_r = (r - mean) / (std + 1e-8)

    advantages = []
    for norm_r_i, length in zip(normalized_r, output_lengths):
        advantages.extend([norm_r_i] * length)
    return advantages


# Process supervision (for each step in the output)
def split_steps(output):
    return output.split('\n')

def reward_process(question, output):
    pass

def adv_process(rewards, output_lengths):
    pass 


def gpro_loss(pi_theta, pi_ref, advantages, epsilon, beta):
   pass

def update_policy(): pass


def train_grpo(
    pi_old, # initial policy model
    r_phi, # reward function
    D, # dataset of questions
    epsilon, # clip parameter
    beta, # KL divergence coefficient
    mu, # number of GRPO optimization steps per batch
    I, # number of iterations
    M, # number of batches per iteration
    G # number of sample outputs per question     
):
    pi_theta = copy_model(pi_old)
    
    for i in range(I):
        for m in range(M):
            D_b = sample_batch(D)

            all_outputs = []
            all_rewards = []
            all_advantages = []

            for q in D_b:
                outputs, output_lengths = generate_outputs(pi_old, q, G)
                rewards = [r_phi(q, o) for o in outputs]
                advantages = adv_outcome(rewards, output_lengths)

                all_outputs.append(outputs)
                all_rewards.append(rewards)
                all_advantages.append(advantages)

            for _ in range(mu):
                pi_theta = update_policy()
    
    return pi_theta 