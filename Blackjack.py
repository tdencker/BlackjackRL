import MonteCarloES as mc
import random
import numpy as np

def dealers_turn(player_sum, dealers_card):
    is_ace = dealers_card == "A"
    sum = dealers_card if not is_ace else 11
    usable_ace = is_ace

    while sum < 17:
        drawn_card = random.choice(deck)
        card_is_ace = drawn_card == "A"
        usable_ace = card_is_ace or usable_ace
        sum += 11 if card_is_ace else drawn_card

        if sum > 21 and usable_ace:
            sum -= 10
            usable_ace = False

    return 1 if sum > 21 else np.sign(player_sum - sum)

def get_state_id(usable_ace, dealers_card, sum):
    dealers_card = 1 if dealers_card == "A" else dealers_card
    return (1 - usable_ace) * 100 + (dealers_card - 1) * 10 + (sum - 12)

def transition_state(state_idx, action):
    usable_ace, dealers_card, sum = states[state_idx]

    if action == HIT:
        drawn_card = random.choice(deck)
        card_is_ace = drawn_card == "A"
        usable_ace = card_is_ace or usable_ace
        sum += 11 if card_is_ace else drawn_card

    if sum > 21 and usable_ace:
        sum -= 10
        usable_ace = False

    if sum > 21 or action == STICK:
        reward = dealers_turn(sum, dealers_card) if sum <= 21 else -1
        next_state = None
    else:
        next_state = get_state_id(usable_ace, dealers_card, sum)
        reward = 0
    return next_state, reward

def print_results(policies):
    import matplotlib.pyplot as plt
    steps_no_ace = []
    for row in policies[:100].reshape(10,10):
        steps_no_ace.append(np.argmax(row == 0) + 11)

    plt.subplot(1,2,1)
    axes = plt.gca()
    axes.set_ylim([10,22])
    plt.step(deck_values, steps_no_ace, where="mid")

    steps_with_ace = []
    for row in policies[100:].reshape(10,10):
        steps_with_ace.append(np.argmax(row == 0) + 11)
    plt.subplot(1,2,2)
    axes = plt.gca()
    axes.set_ylim([10,22])
    plt.step(deck_values, steps_with_ace, where="mid")

    plt.show()

if __name__ == "__main__":
    deck = [item for list in ["A", [x for x in range(2,11)], [10] * 3] for item in list]
    deck_values = sorted(set(deck), key = lambda x : x if type(x) == int else 1)
    states = [ (usable_ace, dealers_card, sum)  
              for usable_ace in [True, False] 
              for dealers_card in deck_values 
              for sum in range(12,22) ]
    STICK = 0
    HIT = 1
    monte_carlo = mc.MonteCarloES(200, 2, transition_state)
    for idx in range(0, 200):
        monte_carlo.policies[idx] = HIT if states[idx][2] >= 20 else STICK
    monte_carlo.run(500000)
    print(monte_carlo.policies[100:].reshape(10,10))
    print()
    print(monte_carlo.policies[:100].reshape(10,10))
    print_results(monte_carlo.policies)