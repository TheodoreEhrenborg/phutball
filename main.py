# Try profiling the code at some point
import copy
import string
import numpy as np
import time

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
# I have had lots of fun dealing with
# formatting problems caused by circle sizes
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
        if array is not None:
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

    def get_ball_moves(self, moves=None, current_name=""):
        """Recursive method. Note that 'self' changes. Also, "Default
        parameter values are evaluated from left to right when the
        function definition is executed", so it's unwise to make
        ~moves~ mutable, because then it gets changed, and Python will
        keep using it as the default.
        See
        https://docs.python.org/3/reference/compound_stmts.html#function-definitions
        """
        #        print(moves)
        if moves is None:
            moves = {}
        if self.ball_at[1] in (-1, LENGTH):
            # We jumped off the endzones and can't make any more jumps
            return moves
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
                # "It is also legal for the ball to leave the board,
                # but only by jumping over a man on the goal line"
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
            moves = new.get_ball_moves(moves, new_name)
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

    def get_all_nearby_moves(self):
        moves = self.get_nearby_man_moves()
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
    current_board.pretty_print_details()
    start_time = time.time()
    while True:
        player = players[num_moves_made % 2]
        move = player.make_move(current_board.copy())
        moves_made.append(move)
        print(moves_made)
        print(
            "Duration of game so far is",
            time.time() - start_time,
            "seconds",
        )
        #        current_board.pretty_print_details()
        current_board = current_board.get_all_moves()[move]
        current_board.pretty_print_details()
        # Previous line makes the move, and returns an error if invalid
        if current_board.ball_at[1] <= 0:
            print("Right has won")
            return
        if current_board.ball_at[1] >= LENGTH - 1:
            print("Left has won")
            return
        num_moves_made += 1


class LocationEvaluator:
    """Returns the normalized position of the ball. The score is close to 1 if the
    ball is near the goal of the player to move. It's 0 in the converse case

    """

    def score(self, board):
        location = board.ball_at[1]
        if location <= 0:
            location = 0
        if location >= LENGTH - 1:
            location = LENGTH - 1
        value = location / (LENGTH - 1)
        # This is near 1 if we're to the east,
        # near the goal Right is protecting
        if board.side_to_move == "Left":
            return value
        else:
            return 1 - value


class NegamaxPlayer:
    def __init__(
        self, depth=1, static_evaluator=LocationEvaluator()
    ):
        self.depth = depth
        self.static_evaluator = static_evaluator

    def score(self, board, depth):
        """Returns the score of the player to move"""
        if board.ball_at[1] in (
            0,
            -1,
            LENGTH - 1,
            LENGTH,
        ):
            # If we've won/lost, we need to stop thinking and
            # accurately report it
            return LocationEvaluator().score(board)
        if depth == 0:
            # Don't keep thinking if we've run out of depth
            self.calls += 1
            return self.static_evaluator.score(board)
        max_score = 0
        possible_moves = board.get_all_nearby_moves()
        for board in possible_moves.values():
            temp_score = self.score(board, depth - 1)
            corrected_score = 1 - temp_score
            # We are the player to move, so we maximize
            # the score of our move
            if corrected_score >= max_score:
                max_score = corrected_score
        return max_score

    def make_move(self, board):
        self.calls = 0
        print(
            "Applying static evaluator to current position:"
        )
        print(self.static_evaluator.score(board))
        max_move = None
        max_score = 0
        print("Initial score is 0")
        possible_moves = board.get_all_nearby_moves()
        for move in possible_moves.keys():
            temp_score = self.score(
                possible_moves[move], self.depth - 1
            )
            corrected_score = 1 - temp_score
            # We are the player to move, so we maximize
            # the score of our move
            if corrected_score >= max_score:
                max_score = corrected_score
                max_move = move
                print(
                    "New best move is",
                    max_move,
                    "which has score of",
                    max_score,
                )
        print(self.calls)
        return max_move


class NegamaxABPlayer:
    def __init__(
        self, depth=1, static_evaluator=LocationEvaluator()
    ):
        self.depth = depth
        self.static_evaluator = static_evaluator

    def score(self, board, depth, alpha, beta):
        """Returns the score of the player to move"""
        if board.ball_at[1] in (
            0,
            -1,
            LENGTH - 1,
            LENGTH,
        ):
            # If we've won/lost, we need to stop thinking and
            # accurately report it
            return LocationEvaluator().score(board)
        if depth == 0:
            # Don't keep thinking if we've run out of depth
            self.calls += 1
            return self.static_evaluator.score(board)
        value = 0
        possible_moves = board.get_all_nearby_moves()
        # Don't reorder the moves because we're not at the top level
        for board in possible_moves.values():
            corrected_score = 1 - self.score(
                board, depth - 1, 1 - beta, 1 - alpha
            )
            # We are the player to move, so we maximize
            # the score of our move
            value = max(corrected_score, value)
            alpha = max(alpha, value)
            if alpha >= beta:
                # We can guarantee a score better than beta,
                # which is the score our opponent can guarantee
                # by not choosing this node. So they won't choose this node.
                if depth > 1:
                    print(
                        "Pruned, depth = ",
                        depth,
                        "alpha = ",
                        alpha,
                        "beta =",
                        beta,
                    )
                break
        return value

    def make_move(self, board):
        self.calls = 0
        print(
            "Applying static evaluator to current position:"
        )
        print(self.static_evaluator.score(board))
        max_move = None
        max_score = 0
        print("Initial score is 0")
        possible_moves = board.get_all_nearby_moves()
        move_list = list(possible_moves.keys())
        if self.depth > 1 and False:
            # Sort the moves using a shallower search
            move_list.sort(
                key=lambda m: 1
                - self.score(
                    possible_moves[m],
                    self.depth - 2,
                    alpha=0,
                    beta=1,
                ),
                reverse=True,
            )
        print(move_list)
        for move in move_list:
            temp_score = self.score(
                possible_moves[move],
                self.depth - 1,
                alpha=0,
                beta=1,
            )
            corrected_score = 1 - temp_score
            # We are the player to move, so we maximize
            # the score of our move
            if corrected_score >= max_score:
                max_score = corrected_score
                max_move = move
                print(
                    "New best move is",
                    max_move,
                    "which has score of",
                    max_score,
                )
            if max_score == 1:
                # This is the alpha-beta break, except at the
                # top level it just means we've won
                break
        print(self.calls)
        return max_move
