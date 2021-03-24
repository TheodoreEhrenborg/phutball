# Try profiling the code at some point
# One Player I could make is one that outputs results of a saved game
import copy
import string
import numpy as np

LENGTH = 19
WIDTH = 15
MAN = "man"
EMPTY = "empty"
BALL = "ball"
DIRECTIONS = {
    "N": np.array((-1, 0)),
    "S": np.array((1, 0)),
    "E": np.array((0, 1)),
    "W": np.array((0, -1)),
    "NW": np.array((-1, -1)),
    "SW": np.array((1, -1)),
    "SE": np.array((1, 1)),
    "NE": np.array((-1, 1)),
}
# I have had lots of fun dealing with formatting problems caused by circle sizes
SMALL_WHITE_CIRCLE = "⚬"  # Looks good on WordPress
SMALL_BLACK_CIRCLE = "•"
WHITE_CIRCLE = "○"  # Looks good on GitHub
BLACK_CIRCLE = "●"


class HumanPlayer:
    def make_move(self, board):
        board.pretty_print_details()
        possible_moves = board.get_all_moves()
        human_move = ""
        while (
            human_move not in possible_moves
            and human_move + " " not in possible_moves
        ):
            human_move = input("Move: ")
        if human_move in possible_moves:
            return human_move
        return human_move + " "


class Board:
    """A virtual board---can be printed out nicely"""

    def __init__(
        self,
        side_to_move="Left",
        moves_made=0,
        array=None,
        ball_at=np.array((7, 9)),
    ):
        self.ball_at = ball_at
        self.side_to_move = side_to_move
        self.moves_made = moves_made
        if array != None:
            self.array = copy.deepcopy(array)
        else:
            self.array = [
                [EMPTY for i in range(LENGTH)]
                for j in range(WIDTH)
            ]
            self.array[ball_at[0]][ball_at[1]] = BALL

    def copy(self):
        return Board(
            side_to_move=self.side_to_move,
            moves_made=self.moves_made,
            array=self.array,
            ball_at=self.ball_at,
        )

    def pretty_string_details(self, small=False):
        if small:
            white_circle = SMALL_WHITE_CIRCLE
            black_circle = SMALL_BLACK_CIRCLE
        else:
            white_circle = WHITE_CIRCLE
            black_circle = BLACK_CIRCLE
        output = ""
        output += "          1111111111\n"
        output += " 1234567890123456789\n"
        for i in range(WIDTH):
            output += string.ascii_uppercase[i]
            for j in range(LENGTH):
                element = self.array[i][j]
                if element == MAN:
                    output += white_circle
                elif element == BALL:
                    output += black_circle
                elif element == EMPTY:
                    output += "+"
                else:
                    raise Exception(
                        "The array contains something it shouldn't: "
                        + str(element)
                    )
            output += "\n"
        output += (
            "Side to move: " + str(self.side_to_move) + "\n"
        )
        output += (
            "Moves made: " + str(self.moves_made) + "\n"
        )
        output += "Ball at: " + str(self.ball_at) + "\n"
        return output

    def pretty_print(self, small=False):
        print(self.pretty_string(small))

    def pretty_print_details(self, small=False):
        print(self.pretty_string_details(small))

    def pretty_string(self, small=False):
        if small:
            white_circle = SMALL_WHITE_CIRCLE
            black_circle = SMALL_BLACK_CIRCLE
        else:
            white_circle = WHITE_CIRCLE
            black_circle = BLACK_CIRCLE
        output = ""
        for i in range(WIDTH):
            for j in range(LENGTH):
                element = self.array[i][j]
                if element == MAN:
                    output += white_circle
                elif element == BALL:
                    output += black_circle
                elif element == EMPTY:
                    output += "+"
                else:
                    raise Exception(
                        "The array contains something it shouldn't: "
                        + str(element)
                    )
            output += "\n"
        return output

    def increment(self):
        if self.side_to_move == "Left":
            self.side_to_move = "Right"
        else:
            self.side_to_move = "Left"
        self.moves_made += 1
        return self

    def get_man_moves(self):
        """Returns a dictionary of all moves that involve
        placing a man. e.g. key = 'B15', value = the future Board"""
        moves = {}
        for i in range(WIDTH):
            for j in range(LENGTH):
                if self.array[i][j] == EMPTY:
                    new = self.copy()
                    new.increment()
                    new.array[i][j] = MAN
                    name = string.ascii_uppercase[i] + str(
                        j + 1
                    )
                    moves[name] = new
        return moves

    def get_nearby_man_moves(self):
        """Like get_man_moves(), except we'd only place a man within 1
        or 2 squares (in a 5x5 box) of an existing piece"""
        moves = {}
        for i in range(WIDTH):
            for j in range(LENGTH):
                if self.array[i][j] == EMPTY:
                    isolated = True
                    for ii in range(i - 2, i + 3):
                        if ii in range(WIDTH):
                            for jj in range(j - 2, j + 3):
                                if jj in range(LENGTH):
                                    if (
                                        self.array[ii][jj]
                                        != EMPTY
                                    ):
                                        isolated = False
                    if not isolated:
                        new = self.copy()
                        new.increment()
                        new.array[i][j] = MAN
                        name = string.ascii_uppercase[
                            i
                        ] + str(j + 1)
                        moves[name] = new
        return moves

    def get_ball_moves(self):
        moves = {}
        current_name = ""
        return self.recursive_get_ball_moves(
            moves, current_name
        )

    # I have to wrap the recursive method in this one,
    # else I get a curious problem with the namespace.

    def recursive_get_ball_moves(self, moves, current_name):
        """Recursive method. Note that 'self' changes"""
        #        print(moves)
        for direction in DIRECTIONS.keys():
            vec = DIRECTIONS[direction]
            new = self.copy()
            jump_length = 0
            while jump_length == 0 or (
                new.is_on_board(end_point)
                and new.array[end_point[0]][end_point[1]]
                == MAN
            ):  # How far can we jump?
                jump_length += 1
                end_point = new.ball_at + jump_length * vec
            #            print(direction, jump_length, end_point)
            if jump_length == 1:
                continue  # We didn't actually jump
            if not new.is_on_board(end_point) and (
                end_point - vec
            )[1] not in (0, LENGTH - 1):
                # "It is also legal for the ball to leave the board, but only by jumping over a man on the goal line"
                continue  # We jumped off the board illegally
            for i in range(jump_length):
                temp = (
                    new.ball_at + i * vec
                )  # Clear out what we jumped over
                new.array[temp[0]][temp[1]] = EMPTY
            if new.is_on_board(end_point):
                # Move the ball
                new.array[end_point[0]][end_point[1]] = BALL
            new.ball_at = (
                end_point  # This may be off the board
            )
            # Now add the move to the dictionary
            new_name = current_name + direction + " "
            #            print(new_name)
            #            print(moves)
            moves[new_name] = new.copy().increment()
            #            print(moves)
            # Now look for jump moves that started with this jump
            moves = new.recursive_get_ball_moves(
                moves, new_name
            )
        #            print(moves)
        return moves

    def is_on_board(self, coordinates):
        return coordinates[0] in range(
            WIDTH
        ) and coordinates[1] in range(LENGTH)

    def get_all_moves(self):
        moves = self.get_man_moves()
        moves.update(self.get_ball_moves())
        return moves


class ListPlayer:
    """Rehashes a game"""

    def __init__(
        self, old_game, quieter=False, small=False
    ):
        self.old_game = old_game
        self.quieter = quieter
        self.small = small

    def make_move(self, board):
        if self.quieter:
            board.pretty_print(self.small)
        else:
            board.pretty_print_details(self.small)
        return self.old_game[board.moves_made]


class PloddingPlayer:
    """Place man, jump, repeat"""

    def make_move(self, board):
        possible_moves = board.get_all_moves()
        if board.side_to_move == "Left":
            if "E " in possible_moves:
                return "E "
            i, j = board.ball_at
            return string.ascii_uppercase[i] + str(j + 2)
        else:
            if "W " in possible_moves:
                return "W "
            i, j = board.ball_at
            return string.ascii_uppercase[i] + str(j)


def run_game(players=[HumanPlayer(), HumanPlayer()]):
    """Conducts a game between two Players, [left, right]"""
    num_moves_made = 0
    moves_made = []
    current_board = Board()
    while True:
        player = players[num_moves_made % 2]
        move = player.make_move(current_board.copy())
        moves_made.append(move)
        print(moves_made)
        #        current_board.pretty_print_details()
        current_board = current_board.get_all_moves()[move]
        # Previous line makes the move, and returns an error if invalid
        if current_board.ball_at[1] <= 0:
            print("Right has won")
            return
        if current_board.ball_at[1] >= LENGTH - 1:
            print("Left has won")
            return
        num_moves_made += 1
