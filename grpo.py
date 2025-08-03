import copy, random


def clone_model(model): 
    cloned = copy.deepcopy(model)
    return cloned

def sample_batch(dataset, batch_size=4):
    return random.sample(dataset, batch_size)

def sample_outputs(policy_model, question, group_size):
    outputs = []
    for _ in range(group_size):
        output = policy_model.run(question)
        outputs.append(output)
    return outputs

def compute_group_advantages(rewards): pass
def update_policy_with_grpo(): pass



def grpo(
    pi_theta_init,  # initial policy model
    r_phi,          # reward function
    D,              # dataset of questions
    epsilon,        # clip parameter
    beta,           # KL divergence coefficient
    mu,             # number of GRPO optimization steps per batch
    I,              # number of outer iterations
    M,              # number of batches per iteration
    G               # number of sampled outputs per question     
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
                outputs = sample_outputs(pi_theta_old, q, G)
                rewards = [r_phi(q, o) for o in outputs]
                advantages = compute_group_advantages(rewards)

                all_outputs.append(outputs)
                all_rewards.append(rewards)
                all_advantages.append(advantages)

            for _ in range(mu):
                pi_theta = update_policy_with_grpo()
    
    return pi_theta 