
from poker.poker_game import PokerGame
from poker.cfr import CFRTrainer

def main():
    num_players = 6
    initial_chips = 1000
    num_ai_players = 5
    small_blind = 5
    big_blind = 10

    game = PokerGame(num_players, initial_chips, num_ai_players, small_blind, big_blind, 100)

    # Train the AI player using the CFR algorithm
    cfr_trainer = CFRTrainer(iterations=10000)
    cfr_trainer.train()

    # Play the game
    game.play_game(num_hands=2)

if __name__ == "__main__":
    main()