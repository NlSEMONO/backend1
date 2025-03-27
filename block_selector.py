import pygame
import random

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 900, 800
BLOCK_SIZE = 30
GRID_COLS = 7  # 7 columns to fit 35 blocks in a 5-row grid
GRID_ROWS = 5
PADDING = 10
BOTTOM_DISPLAY_START = 675  # Y position to display selected blocks
BUTTON_WIDTH, BUTTON_HEIGHT = 120, 50
BUTTON_X, BUTTON_Y = (WIDTH - BUTTON_WIDTH) // 2, HEIGHT - 70  # Center the button

# Colors
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
BLUE = (0, 0, 255)
GREEN = (0, 200, 0)
HIGHLIGHT = (255, 0, 0)

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tetris Block Selector")

# Tetris Shapes (35 Unique Blocks Example)
TETRIS_BLOCKS = [
    [[1, 1, 1, 1]],  # I piece
    [[1, 1], [1, 1]],  # O piece
    [[1, 1, 1], [0, 1, 0]],  # T piece
    [[1, 1, 0], [0, 1, 1]],  # S piece
    [[0, 1, 1], [1, 1, 0]],  # Z piece
    [[1, 1, 1], [1, 0, 0]],  # L piece
    [[1, 1, 1], [0, 0, 1]],  # J piece
]

# Generate 35 random blocks by rotating/flipping existing ones
ALL_BLOCKS = []
for i in range(35):
    block = random.choice(TETRIS_BLOCKS)
    for _ in range(random.randint(0, 3)):  # Rotate randomly
        block = list(zip(*block[::-1]))  # Rotate 90 degrees
    if random.choice([True, False]):  # Flip randomly
        block = [row[::-1] for row in block]
    ALL_BLOCKS.append(block)

# Selection state
selected_blocks = []

def draw_block(surface, block, x, y, color=BLUE):
    """Draws a given Tetris block at a position"""
    for row_idx, row in enumerate(block):
        for col_idx, cell in enumerate(row):
            if cell:
                pygame.draw.rect(surface, color, pygame.Rect(
                    x + col_idx * BLOCK_SIZE, y + row_idx * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE
                ))
                pygame.draw.rect(surface, WHITE, pygame.Rect(
                    x + col_idx * BLOCK_SIZE, y + row_idx * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE
                ), 2)

def draw_grid():
    """Draws the selection grid"""
    for i, block in enumerate(ALL_BLOCKS):
        col = i % GRID_COLS
        row = i // GRID_COLS
        x = PADDING + col * (BLOCK_SIZE * 4 + PADDING)
        y = PADDING + row * (BLOCK_SIZE * 4 + PADDING)
        
        # Highlight if block exists in selected list
        # is_selected = any(b[1] == i for b in selected_blocks)
        # color = HIGHLIGHT if is_selected else BLUE
        draw_block(screen, block, x, y)

def draw_selected_blocks():
    """Displays selected blocks at the bottom"""
    start_x = PADDING
    start_y = BOTTOM_DISPLAY_START
    for block, _ in selected_blocks:
        draw_block(screen, block, start_x, start_y, HIGHLIGHT)
        start_x += BLOCK_SIZE * 4 + PADDING  # Move right for next block
        if start_x + BLOCK_SIZE * 4 > WIDTH:  # Wrap to next row if needed
            start_x = PADDING
            start_y += BLOCK_SIZE * 4 + PADDING

def get_block_at_position(pos):
    """Returns the index of the clicked block or None"""
    mouse_x, mouse_y = pos
    for i in range(35):
        col = i % GRID_COLS
        row = i // GRID_COLS
        x = PADDING + col * (BLOCK_SIZE * 4 + PADDING)
        y = PADDING + row * (BLOCK_SIZE * 4 + PADDING)
        block_width = len(ALL_BLOCKS[i][0]) * BLOCK_SIZE
        block_height = len(ALL_BLOCKS[i]) * BLOCK_SIZE
        if x <= mouse_x <= x + block_width and y <= mouse_y <= y + block_height:
            return i
    return None

def get_selected_block_at_position(pos):
    """Returns the index of the clicked block in the selected list"""
    mouse_x, mouse_y = pos
    start_x = PADDING
    start_y = BOTTOM_DISPLAY_START
    for i, (block, index) in enumerate(selected_blocks):
        block_width = len(block[0]) * BLOCK_SIZE
        block_height = len(block) * BLOCK_SIZE
        if start_x <= mouse_x <= start_x + block_width and start_y <= mouse_y <= start_y + block_height:
            return i  # Return the index in selected_blocks list
        start_x += BLOCK_SIZE * 4 + PADDING  # Move right
        if start_x + BLOCK_SIZE * 4 > WIDTH:  # Wrap to next row
            start_x = PADDING
            start_y += BLOCK_SIZE * 4 + PADDING
    return None

def draw_ok_button():
    """Draws the 'OK' button"""
    pygame.draw.rect(screen, GREEN, (BUTTON_X, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT), border_radius=10)
    font = pygame.font.Font(None, 36)
    text = font.render("OK", True, WHITE)
    text_rect = text.get_rect(center=(BUTTON_X + BUTTON_WIDTH // 2, BUTTON_Y + BUTTON_HEIGHT // 2))
    screen.blit(text, text_rect)

def is_ok_button_clicked(pos):
    """Checks if the 'OK' button was clicked"""
    x, y = pos
    return BUTTON_X <= x <= BUTTON_X + BUTTON_WIDTH and BUTTON_Y <= y <= BUTTON_Y + BUTTON_HEIGHT

# Game Loop
running = True
while running:
    screen.fill(GRAY)

    # Draw all Tetris blocks
    draw_grid()

    # Draw selected blocks at the bottom
    draw_selected_blocks()
    
    # Draw 'OK' button
    draw_ok_button()

    # Handle Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            clicked_index = get_block_at_position(pygame.mouse.get_pos())
            selected_block_index = get_selected_block_at_position(pygame.mouse.get_pos())
            if clicked_index is not None:
                block = ALL_BLOCKS[clicked_index]
                selected_blocks.append((block, clicked_index))  # Add a new selection
            elif selected_block_index is not None:
                selected_blocks.pop(selected_block_index)  # Remove clicked block from bottom list
            elif is_ok_button_clicked(pygame.mouse.get_pos()) and len(selected_blocks) == 3:
                print(f"Confirmed selection: {len(selected_blocks)} blocks")
                selected_blocks = []
                # Extend this to process selections as needed

    pygame.display.flip()

pygame.quit()
