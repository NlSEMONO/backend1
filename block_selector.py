import pygame
import random

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1000, 800
BLOCK_SIZE = 30
GRID_COLS = 7  # 7 columns to fit 35 blocks in a 5-row grid
GRID_ROWS = 5
PADDING = 10

# Colors
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
BLUE = (0, 0, 255)
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
selected_index = 0

def draw_block(surface, block, x, y, color=BLUE):
    """Draws a given Tetris block at a position"""
    for row_idx, row in enumerate(block):
        for col_idx, cell in enumerate(row):
            if cell:
                pygame.draw.rect(surface, color, pygame.Rect(
                    x + col_idx * BLOCK_SIZE, y + row_idx * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE
                ))

def draw_grid():
    """Draws the selection grid"""
    for i, block in enumerate(ALL_BLOCKS):
        col = i % GRID_COLS
        row = i // GRID_COLS
        x = PADDING + col * (BLOCK_SIZE * 4 + PADDING)
        y = PADDING + row * (BLOCK_SIZE * 4 + PADDING)
        color = HIGHLIGHT if i == selected_index else BLUE
        draw_block(screen, block, x, y, color)

# Game Loop
running = True
while running:
    screen.fill(GRAY)

    # Draw all Tetris blocks
    draw_grid()

    # Handle Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                selected_index = (selected_index + 1) % 35
            elif event.key == pygame.K_LEFT:
                selected_index = (selected_index - 1) % 35
            elif event.key == pygame.K_DOWN:
                selected_index = (selected_index + GRID_COLS) % 35
            elif event.key == pygame.K_UP:
                selected_index = (selected_index - GRID_COLS) % 35

    pygame.display.flip()

pygame.quit()
