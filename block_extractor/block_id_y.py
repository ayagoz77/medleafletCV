import numpy as np
from typing import List, Tuple, Dict


class TextBlockIdentifier:
    def __init__(self, y_threshold = 12) -> None:
        self.y_threshold = y_threshold

    def __call__(self, bboxes: List[List[Tuple]]) -> List[Tuple]:
        sorted_bboxes = sorted(bboxes, key=lambda bbox: bbox[1])
        block_ind = []
        current_block = [sorted_bboxes[0]]

        for bbox in sorted_bboxes[1:]:
            if bbox[1] - current_block[-1][1] <= self.y_threshold:
                current_block.append(bbox)
            else:
                block_ind.append(current_block)
                current_block = [bbox]

        block_ind.append(current_block)
        print('blocks', block_ind)
        cluster_y = self.get_y(block_ind)
        return cluster_y
    

    def get_y(self, block_ind):
        results = []
        for block in block_ind:
            if block:
                block_array = np.array(block)
                max_y = np.max(block_array[:, 1])
                min_y = np.min(block_array[:, 1])
                results.append((max_y, min_y))
            else:
                results.append((None, None))
        return results


    def visualize_blocks(self, block_ind):
        plt.figure(figsize=(10, 6))
        for idx, block in enumerate(block_ind):
            x_values = [bbox[0] for bbox in block]
            y_values = [bbox[1] for bbox in block]
            plt.scatter(x_values, y_values, label=f'Block {idx + 1}')

        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.title('Text Blocks Visualization')
        plt.legend()
        plt.grid(True)
        plt.show()