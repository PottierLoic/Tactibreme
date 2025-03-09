import pandas as pd
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

def list_games(df):
    """Splits the dataframe into separate games."""
    df["game_id"] = df["win"].shift(fill_value=0).cumsum()
    return [group for _, group in df.groupby("game_id")]

def count_moves(game_df):
    """Counts the number of moves in a game."""
    return len(game_df)

def get_winner(game_df):
    """Returns the winner of the game."""
    winner_row = game_df[game_df["win"] == 1]
    return winner_row["agent"].iloc[-1] if not winner_row.empty else None

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

def count_retreats_by_agent(game_df):
    """Counts the number of retreats per agent in a game."""
    return game_df.groupby("agent")["retreat_cause"].sum().to_dict()

def count_win(df):
    """Count the number of victorys per color."""
    df["win"] = pd.to_numeric(df["win"], errors="coerce").fillna(0).astype(int)
    win_count_by_color = df.groupby("color")["win"].sum()
    for color, win_count in win_count_by_color.items():
        print(f"[Victory count]\t{color_name(color)}: {win_count}")

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
    Calculates and displays the winrate for each destination position (dest_x, dest_y).
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
            print(f"\t({x}, {y}): {win_rate:.2f}% win")

def calculate_winrate_paw(df):
    """
    Calculates and displays the winrate for each paw that gives win.
    """
    if not {"id_paw", "win"}.issubset(df.columns):
        print("Error: The DataFrame must contain 'id_paw' and 'win' columns.")
        return
    win_rate_paw = df.groupby(["id_paw"])["win"].mean() * 100
    print("[Winrate by paw]")
    if win_rate_paw.empty:
        print("No data available.")
    else:
        for paw, win_rate in win_rate_paw.items():
            print(f"\t{paw_name(paw)}: {win_rate:.2f}% win")

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)
    path = sys.argv[1]
    df = load_data(path)

    print("====================Stats for each games====================")
    games = list_games(df)
    for i, game in enumerate(games):
        print(f"Game {i + 1}:")
        # moves count
        nb_moves = count_moves(game)
        print(f"\t[Total moves]\t{nb_moves}")
        # win
        winner = get_winner(game)
        pos = get_winner_position(game)
        paw = get_winning_paw(game)
        print(f"\t[Winner]\t{color_name(winner)} at {pos} with {paw_name(paw)}")
        # retreat
        total_retreat = count_total_retreats(game)
        retreat_per_agent = count_retreats_by_agent(game)
        print(f"\t[{total_retreat} Retreats]\t{color_name(0)}: {retreat_per_agent.get(0, 0)}, {color_name(1)}: {retreat_per_agent.get(1, 0)}")

    print("====================All games stats====================")
    # victory count + winrate agent
    count_win(df)
    calculate_winrate(df)
    # position winrate (x, y)
    calculate_winrate_pos(df)
    # last paw winrate
    calculate_winrate_paw(df)
    # pile winrate
    # average count moves
    # average retreat

if __name__ == "__main__":
    main()
