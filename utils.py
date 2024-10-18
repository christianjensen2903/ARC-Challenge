def get_column_label(n):
    result = ""
    while n >= 0:
        result = chr(n % 26 + ord("A")) + result
        n = n // 26 - 1
    return result


rgb_lookup = {
    0: (200, 200, 200),  # Nothing (empty space)
    1: (205, 23, 24),  # Red (🔴)
    2: (10, 177, 1),  # Green (🟢)
    3: (0, 89, 209),  # Blue (🔵)
    4: (255, 217, 0),  # Yellow (🟡)
    5: (170, 38, 255),  # Purple (🟣)
    6: (0, 0, 0),  # Grey (⚫️)
    7: (230, 132, 0),  # Orange (🟠)
    8: (255, 255, 255),  # White (⚪️)
    9: (122, 71, 34),  # Brown (🟤)
}
