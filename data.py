from dataclasses import dataclass
from typing import Optional
from typing_extensions import Self

import numpy as np

# generate data for visualization
@dataclass
class Hidden:
    id: int
    x: Optional[float] = 0.
    y: Optional[float] = 0.
    target: Optional['Hidden'] = None

@dataclass
class HiddenGroup:
    id: int
    hiddens: list[Hidden] = None

@dataclass
class Theta:
    id: int
    x: Optional[float] = 0.
    y: Optional[float] = 0.
    target: Optional[Hidden | Self] = None

# Generate data for visualization
def generate_data(num_hiddens: int, num_hidden_groups: int, num_thetas: int):
    # Generate Hidden instances
    hiddens = [Hidden(id=i) for i in range(num_hiddens)]

    # Generate HiddenGroup instances
    hidden_groups = []
    hidden_indices = list(range(num_hiddens))
    np.random.shuffle(hidden_indices)  # Shuffle indices to distribute hiddens randomly

    for i in range(num_hidden_groups):
        # Determine the number of hiddens in this group
        if i < num_hiddens % num_hidden_groups:
            group_size = num_hiddens // num_hidden_groups + 1
        else:
            group_size = num_hiddens // num_hidden_groups

        # Select hiddens for this group
        group_hiddens = [hiddens[hidden_indices.pop()] for _ in range(group_size)]
        hidden_groups.append(HiddenGroup(id=i, hiddens=group_hiddens))

        # Link Hidden instances within the same group to form a tree-like structure
        if len(group_hiddens) > 1:
            for j in range(1, len(group_hiddens)):
                # Randomly select a target for each hidden
                target_index = np.random.choice(range(j))
                group_hiddens[j].target = group_hiddens[target_index]

    # Generate Theta instances
    thetas = [Theta(id=i) for i in range(num_thetas)]

    # Link Theta instances to Hidden instances
    for theta in thetas:
        theta.target = np.random.choice(hiddens)

    return hiddens, hidden_groups, thetas

if __name__ == '__main__':
    # Example usage
    num_hiddens = 10
    num_hidden_groups = 5
    num_thetas = 5

    hiddens, hidden_groups, thetas = generate_data(10, 5, 5)

    # Print the generated data
    print("Hidden Instances:")
    for hidden in hiddens:
        print(hidden)

    print("\nHiddenGroup Instances:")
    for hidden_group in hidden_groups:
        print(hidden_group)

    print("\nTheta Instances:")
    for theta in thetas:
        print(theta)



