import pandas as pd
import numpy as np
import sys

def color_name(color):
    return "blue" if color == 0 else "red"

def paw_name(paw):
    match paw:
        case 0:
            return "Donkey"
        case 1:
            return "Dog"
        case 2:
            return "Cat"
        case 3:
            return "Rooster"
        case _:
            return "UNDEFINED"

def load_data(csv_file):
    """Loads the CSV file and performs necessary conversions."""
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{csv_file}' is empty.")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"Error: File '{csv_file}' is not a valid CSV.")
        sys.exit(1)
    required_columns = {'win', 'retreat_cause', 'dog_pile', 'cat_pile', 'rooster_pile'}
    if not required_columns.issubset(df.columns):
        print("Error: Missing required columns in the CSV file.")
        sys.exit(1)
    df['win'] = df['win'].astype(bool)
    df['retreat_cause'] = df['retreat_cause'].fillna(0).astype(int)
    df[['dog_pile', 'cat_pile', 'rooster_pile']] = df[['dog_pile', 'cat_pile', 'rooster_pile']].astype(bool)
    return df

def list_games(df: pd.DataFrame) -> list:
    """Split the dataframe for each game_id."""
    return [group for _, group in df.groupby("game_id")]

def count_moves(game_df):
    """Counts the number of moves in a game."""
    return len(game_df)

def get_color_winner(game_df):
    """Returns the color winner of the game."""
    winner_row = game_df[game_df["win"] == 1]
    return winner_row["color"].iloc[-1] if not winner_row.empty else None

def get_winner_position(game_df):
    """Returns the position (dest_x, dest_y) of the winner in a game."""
    winner_row = game_df[game_df["win"] == 1]
    if not winner_row.empty:
        dest_x = int(winner_row["dest_x"].iloc[-1])
        dest_y = int(winner_row["dest_y"].iloc[-1])
        return (dest_x, dest_y)
    return None

def get_winning_paw(game_df):
    """Returns the paw of the winning movement."""
    winner_row = game_df[game_df["win"] == 1]
    if not winner_row.empty:
        paw = int(winner_row["id_paw"].iloc[-1])
        return paw
    return None

def count_total_retreats(game_df):
    """Counts the total number of retreats in a game."""
    return game_df["retreat_cause"].sum()

def count_retreats_by_color(game_df):
    """Counts the number of retreats per color in a game."""
    return game_df.groupby("color")["retreat_cause"].sum().to_dict()

def count_win(df):
    """Count the number of victorys per color."""
    df["win"] = pd.to_numeric(df["win"], errors="coerce").fillna(0).astype(int)
    win_count_by_color = df.groupby("color")["win"].sum()
    for color, win_count in win_count_by_color.items():
        print(f"[Victory count]\t{color_name(color)}: {win_count}")

def calculate_winrate_per_agent(df):
    """Calculates and displays the winrate per agent."""
    df_wins = df[df["win"] == 1]
    win_count_by_agent = df_wins["agent"].value_counts()
    total_wins = len(df_wins)
    for id_agent, win_count in win_count_by_agent.items():
        winrate = win_count / total_wins
        print(f"[Agent Winrate]\tAgent {id_agent}: {winrate:.2%}")

def calculate_winrate(df):
    """Calculates and displays the winrate per color."""
    df_wins = df[df["win"] == 1]
    win_count_by_color = df_wins["color"].value_counts()
    total_wins = len(df_wins)
    for color, win_count in win_count_by_color.items():
        winrate = win_count / total_wins
        print(f"[Winrate]\t{color_name(color)}: {winrate:.2%}")

def calculate_winrate_pos(df):
    """
    Calculates and displays the winrate for each destination position.
    """
    if not {"dest_x", "dest_y", "win"}.issubset(df.columns):
        print("Error: The DataFrame must contain 'dest_x', 'dest_y', and 'win' columns.")
        return
    win_rate_by_position = df.groupby(["dest_x", "dest_y"])["win"].mean() * 100
    print("[Winrate by position]")
    if win_rate_by_position.empty:
        print("No data available.")
    else:
        for (x, y), win_rate in win_rate_by_position.items():
            print(f"\t({x}, {y}): {win_rate:.2%} win")

def paw_usage_percentage(game_df, buffer):
    """Display the number of moves and percentage of use of each piece for one game."""
    total_moves = len(game_df)
    usage_counts = game_df["id_paw"].value_counts()
    usage_percentages = (usage_counts / total_moves) * 100
    tmp_buff = [0,0,0,0]
    print(f"\t[Paw usage]")
    for paw_id, count in usage_counts.items():
        tmp_buff[paw_id] = count
        print(f"\t\t{paw_name(paw_id)}: {count} moves ({usage_percentages[paw_id]:.2f}%)")
    print(tmp_buff)
    buffer.append(tmp_buff)

def calculate_winrate_paw(df):
    """
    Calculates and displays the percentage of wins attributed to each paw.
    """
    if not {"id_paw", "win"}.issubset(df.columns):
        print("Error: The DataFrame must contain 'id_paw' and 'win' columns.")
        return
    win_counts = df[df["win"] == 1]["id_paw"].value_counts(normalize=True) * 100
    print("[Percentage of total wins by paw]")
    if win_counts.empty:
        print("No wins recorded.")
    else:
        for paw, percentage in win_counts.items():
            print(f"\t{paw_name(paw)}: {percentage:.2f}% wins")

def get_win_type(df):
    """
    Determines how the game was won, which animals were moved during the last moves.
    """
    win_row = df[df["win"] == 1]
    buffer = []
    buffer.append(paw_name(win_row['id_paw'].values[0]))
    if win_row["dog_pile"].values[0]:
        buffer.append(paw_name(1))
    if win_row["cat_pile"].values[0]:
        buffer.append(paw_name(2))
    if win_row["rooster_pile"].values[0]:
        buffer.append(paw_name(3))
    return ', '.join(buffer)

def average_moves(df) -> float:
    """Return the average number of moves."""
    turns_per_game = df.groupby("game_id").size()
    return turns_per_game.mean()

def average_paw_usage(buffer):
    array = np.array(buffer)
    column_means = np.mean(array, axis=0)
    for paw_id in range(array.shape[1]):
        print(f"\t{paw_name(paw_id)}: {column_means[paw_id]:.2f} moves")

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)
    path = sys.argv[1]
    df = load_data(path)
    retreat_buffer = []
    paw_usage_buffer = []
    print("====================Stats for each games====================")
    games = list_games(df)
    for i, game in enumerate(games):
        print(f"Game {i + 1}:")
        nb_moves = count_moves(game)
        print(f"\t[Total moves]\t{nb_moves}")
        winner = get_color_winner(game)
        pos = get_winner_position(game)
        paw = get_winning_paw(game)
        print(f"\t[Winner]\t{color_name(winner)} at {pos} with {paw_name(paw)}")
        total_retreat = count_total_retreats(game)
        retreat_per_color = count_retreats_by_color(game)
        retreat_buffer.append(total_retreat)
        print(f"\t[{total_retreat} Retreats]\t{color_name(0)}: {retreat_per_color.get(0, 0)}, {color_name(1)}: {retreat_per_color.get(1, 0)}")
        paw_usage_percentage(game, paw_usage_buffer)
        print(f"\t[Moved pile]\t{get_win_type(game)}")

    print("====================All games stats====================")
    count_win(df)
    calculate_winrate(df)
    calculate_winrate_pos(df)
    calculate_winrate_paw(df)
    print(f"[Avg moves]\t{average_moves(df)}")
    print(f"[Avg retreats]\t{np.mean(retreat_buffer)}")
    print(f"[Avg paw usage]")
    average_paw_usage(paw_usage_buffer)
    print("====================AI perfs====================")
    calculate_winrate_per_agent(df)

if __name__ == "__main__":
    main()
