from dataclasses import dataclass
from typing import Optional
from typing_extensions import Self

import numpy as np

# generate data for visualization
@dataclass
class Hidden:
    id: int
    target: Optional['Hidden'] = None

    def __hash__(self):
        return hash(self.id)

@dataclass
class HiddenGroup:
    id: int
    hiddens: list[Hidden] = None

@dataclass
class Theta:
    id: int
    target: Optional[Hidden | Self] = None

    def __hash__(self):
        return hash(self.id)

def is_cycle(hidden, target):
    """
    检查是否形成环。

    :param hidden: 当前的 Hidden 实例
    :param target: 目标 Hidden 实例
    :return: 是否形成环
    """
    visited = set()
    while target is not None:
        if target == hidden:
            return True
        if target in visited:
            return True
        visited.add(target)
        target = target.target
    return False

# Generate data for visualization
def generate_data(num_hiddens: int, num_hidden_groups: int, num_thetas: int, seed: int=-1):
    np.random.seed(seed)
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
        max_attempts = 10
        if len(group_hiddens) > 1:
            for j in range(1, len(group_hiddens)):
                # Randomly select a target for each hidden
                target = np.random.choice(group_hiddens[:j])
                attempts = 0
                while attempts < max_attempts:
                    if not is_cycle(group_hiddens[j], target):
                        break
                    target = np.random.choice(group_hiddens[:j])
                    attempts += 1

                if attempts >= max_attempts:
                    target = None

                group_hiddens[j].target = target

    # Generate Theta instances
    thetas = [Theta(id=i) for i in range(num_thetas)]

    # Link Theta instances to Hidden instances
    for theta in thetas:
        random_num = np.random.rand()
        if random_num > 0.5:
            theta.target = np.random.choice(hiddens)
        elif random_num < 0.4:
            target_index = np.random.choice(range(num_thetas))
            while target_index == theta.id:
                target_index = np.random.choice(range(num_thetas))
            theta.target = thetas[target_index]
        else:
            theta.target = None

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



