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

def count_win(df):
    """Count the number of victorys per color."""
    df["win"] = pd.to_numeric(df["win"], errors="coerce").fillna(0).astype(int)
    win_count_by_color = df.groupby("color")["win"].sum()
    for color, win_count in win_count_by_color.items():
        print(f"[Victory count]\t{color_name(color)}: {win_count}")

def calculate_win_rate(df):
    """Calculates and displays the win rate per color."""
    df_wins = df[df["win"] == 1]
    win_count_by_color = df_wins["color"].value_counts()
    total_wins = len(df_wins)
    for color, win_count in win_count_by_color.items():
        winrate = win_count / total_wins
        print(f"[Winrate]\t{color_name(color)}: {winrate:.2%}")

# def calculate_win_rate_pos(df):
#     """Calculates and displays the win rate per position."""
#     if not {'dest_x', 'dest_y'}.issubset(df.columns):
#         print("Error: Position columns are missing.")
#         return
#     position = df.groupby(['dest_x', 'dest_y'])['win'].mean() * 100
#     print("Win rate by position:")
#     print(position.to_string(), "\n")

# def calculate_move_paw_rate(df):
#     """Calculates and displays the movement rate per paw."""
#     if not {'color', 'id_paw'}.issubset(df.columns):
#         print("Error: Movement-related columns are missing.")
#         return
#     paw_rate = df.groupby('color')['id_paw'].value_counts(normalize=True) * 100
#     print("Movement rate by paw:")
#     print(paw_rate.to_string(), "\n")

# def average_number_moves(df):
#     """Calculates and displays the average number of moves per game."""
#     try:
#         moves_per_game = df[df['win']].index.to_series().diff().fillna(0).tolist()
#         avg_moves = sum(moves_per_game) / len(moves_per_game) if moves_per_game else 0
#         print(f"Average number of moves per game: {avg_moves}\n")
#     except ZeroDivisionError:
#         print("Error: No games recorded.")

# def average_number_retreats(df):
#     """Calculates and displays the average number of retreats per game."""
#     try:
#         retreat_counts = df.groupby(df['win'].cumsum())['retreat_cause'].sum()
#         avg_retreats = retreat_counts.mean() if not retreat_counts.empty else 0
#         print(f"Average number of retreats per game: {avg_retreats}\n")
#     except Exception as e:
#         print(f"Error calculating retreats: {e}")

# def win_movement_compo(df):
#     """Displays the composition of winning movements."""
#     winning_compositions = df[df['win']][['dog_pile', 'cat_pile', 'rooster_pile']]
#     print("Win movement composition:")
#     print(winning_compositions.astype(int).to_string(index=False), "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)
    path = sys.argv[1]
    df = load_data(path)
    count_win(df)
    calculate_win_rate(df)
    # calculate_win_rate_pos(df)
    # calculate_move_paw_rate(df)
    # average_number_moves(df)
    # average_number_retreats(df)
    # win_movement_compo(df)

if __name__ == "__main__":
    main()
