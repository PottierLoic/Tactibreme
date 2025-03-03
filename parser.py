import pandas as pd

def calculate_win_rate(csv_file):
    file = pd.read_csv(csv_file)
    file['win'] = file['win'].astype(bool)
    win_rate = file.groupby('agent')['win'].mean() * 100
    for agent, value in win_rate.items():
        print(f"Win rate by agent :\nAgent {agent}: {value}%\n")
    return

def calculate_win_rate_pos(csv_file):
    file = pd.read_csv(csv_file)
    file['win'] = file['win'].astype(bool)
    position = file.groupby(['dest_x', 'dest_y'])['win'].mean() * 100
    for (x,y), value in position.items():
        print(f"Win rate by position :\nPosition {x}, {y}: {value}%\n")
    return

def calculate_move_paw_rate(csv_file):
    file = pd.read_csv(csv_file)
    paw_rate = file.groupby('color')['id_paw'].value_counts(normalize=True) * 100
    for id, value in paw_rate.items():
        print(f"Movement rate by paw :\n{id}: {value}%")
    return
 
def average_number_moves(csv_file):
    file = pd.read_csv(csv_file)
    list_games = []
    number_moves = 0
    for id in file['win']:
        number_moves = number_moves + 1
        if id == 1:
            list_games.append(number_moves)
            number_moves = 0
    print("Average number of moves per game :", sum(list_games)/len(list_games))
    return 

def average_number_retreats(csv_file):
    file = pd.read_csv(csv_file)
    list_games = []
    number_retreats = 0
    for id in file['win']:
        for retreat in file['retreat_cause']:
            if retreat == 1:
                number_retreats = number_retreats + 1
            if id == 1:
                list_games.append(number_retreats)
                number_retreats = 0
    print("Average number of retreats per game :", sum(list_games)/len(list_games))
    return 

def win_movement_compo(csv_file):
    file = pd.read_csv(csv_file)
    winning_compositions = file[file['win'] == 1][['dog_pile', 'cat_pile', 'rooster_pile']]
    for _, row in winning_compositions.iterrows():
        print(f"Win movement composition: \nDog: {bool(row['dog_pile'])}, \nCat: {bool(row['cat_pile'])}, \nRooster: {bool(row['rooster_pile'])}")
    return

if __name__ == "__main__":
    csv_path = "./csv/train_rec_3_model1.csv"
    calculate_win_rate(csv_path)
    calculate_move_paw_rate(csv_path)
    calculate_win_rate_pos(csv_path)
    average_number_moves(csv_path)
    average_number_retreats(csv_path)
    win_movement_compo(csv_path)