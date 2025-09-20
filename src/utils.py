from typing import List, Tuple
import seaborn as sns

def rgb2hex(r, g, b) -> str:
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

def hex2rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_colors(n: int, palette: str = "hls") -> List[str]:
    colors: List[Tuple[float]] = sns.color_palette(palette=palette, n_colors=n)
    colors_rgb: List[List[int]] = []
    for color in colors:
        norm_color: List[int] = [round(x * 255) for x in color]
        colors_rgb.append(norm_color)
    colors_hex: List[str] = [rgb2hex(*color) for color in colors_rgb]
    return colors_hex

def combine_rgb(rgb_colors):
    num_colors = len(rgb_colors)
    if num_colors == 0:
        return (0, 0, 0)
    
    avg_r = sum(c[0] for c in rgb_colors) // num_colors
    avg_g = sum(c[1] for c in rgb_colors) // num_colors
    avg_b = sum(c[2] for c in rgb_colors) // num_colors
    return (avg_r, avg_g, avg_b)
