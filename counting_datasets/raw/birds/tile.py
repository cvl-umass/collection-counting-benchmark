import cv2
import torch
import os
from collections import defaultdict


def split_image(image, tile_size):
    """Splits an image into tiles of given size.

    Args:
        image (numpy.ndarray): The image to split.
        tile_size (tuple): The desired size of each tile (width, height).

    Returns:
        list: A list of tiles as numpy arrays.
    """

    height, width, _ = image.shape
    tiles = []

    for y in range(0, height, tile_size[1]):
        for x in range(0, width, tile_size[0]):
            tile = image[y : y + tile_size[1], x : x + tile_size[0]]
            tiles.append(tile)

    return tiles


if __name__ == "__main__":
    fname = "_DSC5295.tif"
    image = cv2.imread(f"{fname}")
    tile_width = 160  # 200 for _DSC5214
    tile_height = 160

    tiles = split_image(image, (tile_width, tile_height))
    p = torch.randperm(len(tiles), generator=torch.Generator().manual_seed(43)).tolist()

    urls = defaultdict(list)
    # Save tiles
    for i, tile in enumerate(tiles):
        urls[p.index(i) // 50].append(
            f'https://doppler.cs.umass.edu/tiles/tiles{fname[:fname.rfind(".")]}/{p.index(i)//50}/tile_{i}.jpg'
        )
        os.makedirs(
            f"tiles/tiles{fname[:fname.rfind('.')]}/{p.index(i)//50}", exist_ok=True
        )
        cv2.imwrite(
            f"tiles/tiles{fname[:fname.rfind('.')]}/{p.index(i)//50}/tile_{i}.jpg", tile
        )
    # os.makedirs(f'tasks/tiles{fname[:fname.rfind(".")]}', exist_ok=True)
    # for k in urls:
    #     with open(f'tasks/tiles{fname[:fname.rfind(".")]}/{k}.txt', "w") as f:
    #         f.write("\n".join(urls[k]))
    print()
