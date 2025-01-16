# Model Overview

## Input Representation

The board is represented as a 5x5 grid with an 8-channel vector for each tile:
- **Channels**:
  - First 4 channels: Presence of blue paws (`blue_donkey, blue_dog, blue_cat, blue_rooster`).
  - Next 4 channels: Presence of red paws (`red_donkey, red_dog, red_cat, red_rooster`).
- **Example Encoding**:
  - A tile with a blue cat and a red donkey:
    ```
    [0, 0, 1, 0, 1, 0, 0, 0]
    ```
  - An empty tile:
    ```
    [0, 0, 0, 0, 0, 0, 0, 0]
    ```
- **Input Shape**:
  - For a single board: `(8, 5, 5)` (channels, rows, columns).
  - For a batch of boards: `(batch_size, 8, 5, 5)`.


## Output Representation

The output represents possible moves using a flattened encoding of paw-destination pairs:
- **Output Size**: `4 x 25 = 100`
  - 4 possible paws (per player).
  - 25 possible destinations (5x5 board).
- **Flattened Indexing**:
  - Index `i` represents the move:
    ```
    paw_index = i // 25
    destination = (i % 25) // 5, (i % 25) % 5
    ```

- **Masking**:
  - A binary mask is applied to ensure only valid paw-destination pairs are considered:
    - Invalid pairs are set to `-inf`.
    - Valid moves are weighted by the network's predictions.
  - Example:
    ```
    Mask = [0, 1, 0, ..., 1]  # 1 for valid moves, 0 for invalid
    ```

## Neural Network Architecture

The neural network processes the input board and outputs a prediction for paw-destination pairs:
- **Input**: `(batch_size, 8, 5, 5)`
- **Architecture**:
  - **Convolutional Layers**: Extract spatial features and relationships between tiles.
  - **Fully Connected Layers**: Output scores for all 100 possible paw-destination pairs.
- **Output Shape**: `(batch_size, 100)`

### Network Layers
1. **Convolutional Layers**:
   - Conv2D(8 → 32) → ReLU
   - Conv2D(32 → 64) → ReLU
   - Conv2D(64 → 64) → ReLU
2. **Fully Connected Layers**:
   - Flatten
   - Linear(5 * 5 * 64 → 128) → ReLU
   - Linear(128 → 100)

