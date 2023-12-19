import pygame
import random
import numpy as np
import cv2
import time

import tensorflow as tf
from tensorflow import keras

NONE = 0
JUMP = 1
FALL = 2

### GAME CLASSES ###
class Player:
    def __init__(self, x, y, image_path, size, max_jump_height, brain):
        super().__init__()
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, size)  # Resize the image
        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = x, y
        self.initial_y = y  # Save the initial y position
        self.brain = brain
        self.is_jumping = False
        self.is_falling_fast = False
        self.max_jump_height = max_jump_height
        self.jump_speed = self.calculate_jump_speed()
        self.fall_speed_increment = 3  # Base value for how fast the player falls
        self.is_alive = True
        self.score = 0
        self.n_jumps = 0
        self.train_frame = None

    def calculate_jump_speed(self):
        # Calculate the initial jump speed based on the maximum jump height
        return (2 * self.max_jump_height) ** 0.5

    def jump(self):
        self.n_jumps += 1
        if not self.is_jumping:
            self.is_jumping = True
            self.is_falling_fast = False  # Reset fast falling state when jumping
            self.current_jump_speed = self.jump_speed
            self.fall_speed_increment = 1  # Reset to normal fall speed

    def fast_fall(self):
        # Start fast falling
        self.is_falling_fast = True

    def move(self):

        if self.is_jumping:
            # Update the y position based on the current jump speed
            self.rect.y -= self.current_jump_speed
            # Adjust the gravity effect
            gravity_effect = self.fall_speed_increment if not self.is_falling_fast else 6
            self.current_jump_speed -= gravity_effect

            # Check if the player has reached the ground
            if self.rect.y > self.initial_y:
                self.rect.y = self.initial_y
                self.is_jumping = False
                self.is_falling_fast = False  # Reset fast falling state when reaching ground
                self.fall_speed_increment = 1  # Reset fall speed increment

        # Ensure the player never goes below the initial y position
        if self.rect.y > self.initial_y:
            self.rect.y = self.initial_y
            self.is_jumping = False
            self.is_falling_fast = False  # Reset fast falling state
            self.fall_speed_increment = 1  # Reset fall speed increment

class Obstacle(pygame.sprite.Sprite):
    def __init__(self, x, y, image_path, size):
        super().__init__()
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, size)  # Resize the image
        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = x, y

    def move(self):
        self.rect.x -= 5

class Background(pygame.sprite.Sprite):
    def __init__(self, image_path, width, height):
        super().__init__()
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect()

### AUXILIARY CLASSES ###
        
def get_prepared_frame(game_size, frame_size, background, obstacles, player):
    # Create a new surface
    synthetic_surface = pygame.Surface(game_size)

    # Render the background
    synthetic_surface.blit(background.image, background.rect)

    # Render obstacles and player
    for obstacle in obstacles:
        synthetic_surface.blit(obstacle.image, obstacle.rect)
    synthetic_surface.blit(player.image, player.rect)

    # Convert surface to an array and process it
    frame = pygame.surfarray.array3d(synthetic_surface)
    frame = np.transpose(frame, (1, 0, 2))  # Correct the orientation
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR

    # The rest of the processing remains the same
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, frame_size)
    normalized_frame = resized_frame / 255.0

    expanded_frame = np.expand_dims(normalized_frame, axis=-1)  # Add channel dimension
    expanded_frame = np.expand_dims(expanded_frame, axis=0)     # Add batch dimension

    return expanded_frame

### CONTROL CLASSES ###

def get_action(frame, model, print_probs = False):

    # Make the prediction
    probs = model.predict(frame, verbose=0)

    # Convert the prediction to an action (we can choose arbitarely which class is each action)
    max_index = np.argmax(probs[0])

    if print_probs:
        print(probs)
    
    return max_index

### MAIN LOOP ###

def run_game(sleep_time, model_list):

    # Initialize the game
    pygame.init()
    start_time = pygame.time.get_ticks()  # Get the start time
    win_width, win_height = 400, 400
    ground_height = win_height - 75
    win = pygame.display.set_mode((win_width, win_height))

    # Create players, background, and ground instances
    player_list = []
    for model in model_list:
        player = Player(50, ground_height - 15, './data/grinch.png', (50, 50), 150, model)

        # Create a list of players (they have an initial score)
        player_list.append(player)

    background = Background('./data/background.png', win_width, win_height)
    font = pygame.font.SysFont('arial', 18)  # Font name and size

    obstacles = []
    obs_margin = 200
    obs_size = (35, 35)
    run = True
    alive_players = len(player_list)

    while run:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Move player
        for player in player_list:
            if player.is_alive:
                player.move()

        # Create new obstacles
        if random.randint(1, 20) == 1:

            if not len(obstacles) == 0:
                if win_width - obstacles[-1].rect.x > obs_margin:
                    new_obs1 = Obstacle(win_width, ground_height, './data/candy.png', obs_size)
                    obstacles.append(new_obs1)
            else:
                new_obs1 = Obstacle(win_width, ground_height, './data/candy.png', obs_size)
                obstacles.append(new_obs1)

        # Move and remove obstacles
        for ob in obstacles[:]:
            ob.move()
            if ob.rect.x < -ob.rect.width:
                obstacles.remove(ob)
                for player in player_list:
                    if player.is_alive:
                        player.score += 1

        # Collision detection
        for ob in obstacles:
            for player in player_list:
                if player.is_alive and player.rect.colliderect(ob.rect):
                    player.is_alive = False
                    alive_players -= 1

        # Drawing
        win.blit(background.image, background.rect)
        for player in player_list:
            if player.is_alive:
                win.blit(player.image, player.rect)
        for ob in obstacles:
            win.blit(ob.image, ob.rect)

        # Display the number of alive players
        text = "Alive Players: " + str(alive_players)
        text_surface = font.render(text, True, (255, 255, 255))  # Text, anti-aliasing, color (RGB)
        win.blit(text_surface, (250, 20))  # Position (X, Y) where the text is drawn
        pygame.display.update()

        # Generate a game frame
        for player in player_list:
            prepared_frame = get_prepared_frame((400, 400), (50, 50), background, obstacles, player)
            player.frame = prepared_frame

        # Get the control action and execute it
        for player in player_list:
            if player.is_alive:
                action = get_action(player.frame, player.brain, False)
                if action == JUMP:
                    player.jump()
                elif action == FALL:
                    player.fast_fall()
        
        # Check exit condition
        if alive_players == 0:
            run = False
        
        # Check for maximum time of 2 minutes
        current_time = pygame.time.get_ticks()
        if current_time - start_time > 120000:  # 120000 milliseconds = 2 minutes
            run = False

        time.sleep(sleep_time)

    pygame.quit()

    # Generate a list of scores
    results = []
    for player in player_list:
        results.append((player.score, player.n_jumps))

    return results

def main():
    
    model = keras.models.load_model('models/stable50/v1.keras')
    run_game(0, [model])

if __name__ == "__main__":
    main()