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

        if sum >= 17 and usable_ace:
            sum -= 10
            usable_ace = False

    return 1 if sum > 21 else np.sign(player_sum - sum)

def transition_state(state_idx, action_idx):
    usable_ace, dealers_card, sum = states[state_idx]

    if action_idx == HIT:
        drawn_card = random.choice(deck)
        card_is_ace = drawn_card == "A"
        usable_ace = card_is_ace or usable_ace
        sum += 11 if card_is_ace else drawn_card

    if sum > 21 and usable_ace:
        sum -= 10
        usable_ace = False

    if sum > 21 or action_idx == STICK:
        reward = dealers_turn(sum, dealers_card) if sum <= 21 else -1
        next_state = None
    else:
        next_state = (usable_ace, dealers_card, sum)
        reward = 0
    return next_state, reward

if __name__ == "__main__":
    deck = [item for list in ["A", [x for x in range(2,11)], [10] * 3] for item in list]
    states = [ (usable_ace, dealers_card, sum)  
              for usable_ace in [True, False] 
              for dealers_card in sorted(set(deck), key = lambda x : x if type(x) == int else 1) 
              for sum in range(12,22) ]
    STICK = 0
    HIT = 1
    monte_carlo = mc.MonteCarloES(200, 2, transition_state)
    monte_carlo.run(1000)
    print(monte_carlo.getPolicies())