# I could not get this to work as a jupyter notebook, so it's gonna be a python file
# I'll add screenshots of the output to the appendix of the report
import sys
import os
from itertools import count
import torch
from torch.distributions import Categorical
import torch.multiprocessing as mp
from Wordle import WordleEnv
from models.AsyncActorCritic import Actor, Critic

def thread(thread_id, global_actor, global_critic, optimizer_actor, optimizer_critic, env_init, action_size, num_episodes, gamma, lock, device="cpu"):
    print(f"[Thread {thread_id}] Started")
    try: # I had a lot of issues getting the threads to work, so I'm gonna leave the try/except debug statements
        env = env_init()
        local_actor = Actor(env.state_size, action_size).to(device)
        local_critic = Critic(env.state_size, action_size).to(device)
        local_actor.load_state_dict(global_actor.state_dict())
        local_critic.load_state_dict(global_critic.state_dict())
        for episode in range(num_episodes):
            state = env.reset()
            state = torch.FloatTensor(state).to(device).unsqueeze(0)
            log_probs = []
            values = []
            rewards = []
            entropies = []
            episode_reward = 0 # Track cumulative reward per episode instead of letting env reset reward each step()
            for counts in count():
                logits = local_actor(state)
                mask = torch.full((1, action_size), -float('inf'), device=device) # Make a new mask
                for idx in env.available_actions:
                    mask[0, idx] = 0
                masked_logits = logits + mask
                dist = Categorical(logits=masked_logits)
                action = dist.sample() # Functionally the same as select_action_actor() but easier to read
                value = local_critic(state)
                next_state, reward, done, _ = env.step(action.item())
                episode_reward += reward
                next_state = torch.FloatTensor(next_state).to(device).unsqueeze(0)
                log_probs.append(dist.log_prob(action))
                values.append(value)
                rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
                entropies.append(dist.entropy()) # To be added to loss later
                state = next_state
                if done:
                    break
            next_value = local_critic(state)
            returns = []
            next_return = next_value.detach()
            for r in reversed(rewards):
                next_return = r + gamma * next_return
                returns.insert(0, next_return)
            returns = torch.cat(returns).detach() # for advantage
            values = torch.cat(values) # for advantage
            log_probs = torch.cat(log_probs) # for loss
            entropies = torch.cat(entropies) # for loss
            advantage = returns - values # calc advantage
            actor_loss = -(log_probs * advantage.detach()).mean() - 0.01 * entropies.mean() # entropies encourage exploration
            critic_loss = advantage.pow(2).mean()
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            for local_param, global_param in zip(local_actor.parameters(), global_actor.parameters()):
                global_param._grad = local_param.grad # Share gradients with ...
            for local_param, global_param in zip(local_critic.parameters(), global_critic.parameters()):
                global_param._grad = local_param.grad # ... global versions
            with lock: # Only one thread can update the global optimizers at a time
                optimizer_actor.step()
                optimizer_critic.step()
            local_actor.load_state_dict(global_actor.state_dict())
            local_critic.load_state_dict(global_critic.state_dict())
            if thread_id == 0 and episode % 100 == 0: #Only one thread prints
                print(f"[Thread {thread_id}] Episode: {episode}/{num_episodes}, Attempts: {env.attempts}, Reward: {episode_reward}")
    except Exception as e: # On the off chance something fails
        import traceback
        print(f"[Thread {thread_id}] Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.stderr.flush() # Force stderr output to print

def make_env(): # function definition so that the threads can make their own environments
    return WordleEnv()

if __name__ == '__main__':
    mp.set_start_method('spawn') # Start method = new python process
    num_episodes = 1000
    gamma = 0.99 # Very close to 1 to prioritize future rewards
    env = WordleEnv()
    state_size = env.state_size
    action_size = env.action_size
    device = torch.device("cpu")  # Use CPU for multiprocessing
    global_actor = Actor(state_size, action_size).to(device)
    global_critic = Critic(state_size, action_size).to(device)
    global_actor.share_memory()
    global_critic.share_memory()
    optimizer_actor = torch.optim.Adam(global_actor.parameters(), lr=0.001)
    optimizer_critic = torch.optim.Adam(global_critic.parameters(), lr=0.001)
    lock = mp.Lock() # Lock for syncing threads
    processes = []
    for thread_id in range(os.cpu_count()): # Automatically make an agent for each cpu core
        p = mp.Process(
            target=thread,
            args=(thread_id, global_actor, global_critic, optimizer_actor, optimizer_critic,
                    make_env, action_size, num_episodes, gamma, lock)
        )
        p.start()
        print(p)
        processes.append(p)
    for p in processes:
        p.join() # Wait for threads to finish
    torch.save(global_actor.state_dict(), "a3c_actor.pth")
    print("~~~ TRAINING COMPLETE ~~~")

    env = WordleEnv()
    state_size = env.state_size
    action_size = env.action_size
    actor = Actor(state_size, action_size).to(device)
    actor.load_state_dict(torch.load("a3c_actor.pth", map_location=device, weights_only=True)) # weights_only=True or it throws a huge warning each time
    actor.eval()

    def select_action_actor(state, available_actions, action_size, logits=None, device="cpu"):
        with torch.no_grad():
            if logits is None:
                logits = actor(state)
            mask = torch.full((1, action_size), -float('inf'), device=device)
            for idx in available_actions:
                mask[0, idx] = 0
            masked_logits = logits + mask
            masked_distribution = torch.distributions.Categorical(logits=masked_logits)
            return masked_distribution.sample().view(1, 1).squeeze()

    #    
    # Testing with exploration
    #

    total_attempts = 0
    correct_guesses = 0
    no_test_trials = 1000
    print("~~~ BEGINNING TESTING ~~~")

    for episode in range(no_test_trials):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action_actor(state, env.available_actions, env.action_size, device=device)
            observation, reward, done, _ = env.step(action.item())
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            state = next_state
            total_attempts += 1
            if done:
                if reward == 10:
                    correct_guesses += 1
                break
    success_rate = correct_guesses / no_test_trials
    average_attempts = total_attempts / no_test_trials
    print(f"Trials: {no_test_trials}, Success rate: {success_rate:.2f}, Average number of attempts: {average_attempts:.2f}")

    #
    # Testing with SALET start
    #

    env = WordleEnv()
    state_size = env.state_size
    action_size = env.action_size
    actor = Actor(state_size, action_size).to(device)
    actor.load_state_dict(torch.load("a3c_actor.pth", map_location=device, weights_only=True)) # weights_only=True or it throws a huge warning each time
    actor.eval()
    total_attempts = 0
    correct_guesses = 0
    no_test_trials = 1000
    print("~~~ BEGINNING TESTING 2 ~~~")

    for episode in range(no_test_trials):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = torch.tensor([[345]], device=device, dtype=torch.long) #Salet start.
        for t in count():
            observation, reward, done, _ = env.step(action.item())
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            state = next_state
            total_attempts += 1
            if done:
                if reward == 10:
                    correct_guesses += 1
                break
            action = select_action_actor(state, env.available_actions, env.action_size, device=device)
    success_rate = correct_guesses / no_test_trials
    average_attempts = total_attempts / no_test_trials
    print(f"Trials with SALET start: {no_test_trials}, Success rate: {success_rate:.2f}, Average number of attempts: {average_attempts:.2f}")