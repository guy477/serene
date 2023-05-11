
from poker.poker_game import PokerGame
from poker.cfr import CFRTrainer

def main():
    num_players = 6
    num_ai_players = 6

    # pot relative bet-sizings for preflop, flop, turn, and river
    bet_sizing = [(1, 3), (.33, .70), (.40, .82, 1.2), (.75, 1.2, 2)]

    initial_chips = 1000
    small_blind = 5
    big_blind = 10

    num_iterations = 33
    cfr_depth = 8
    cfr_realtime_depth = 4
    

    game = PokerGame(num_players, initial_chips, num_ai_players, small_blind, big_blind, bet_sizing, num_iterations, cfr_depth, cfr_realtime_depth)

    # Train the AI player using the CFR algorithm
    # cfr_trainer = CFRTrainer(iterations=num_iterations, num_players = num_players, initial_chips = initial_chips, small_blind=small_blind, big_blind=big_blind)
    # cfr_trainer.train()

    # Play the game
    game.play_game(num_hands=1)
    print('\n\n')

if __name__ == "__main__":
    main()