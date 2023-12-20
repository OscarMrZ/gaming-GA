import pygame
import random, time



FPS = 60
clock = pygame.time.Clock()

# Screen information
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 500
FLOOR_HEIGHT = 10
BLUE  = (0, 0, 255)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Game information
JUMP_SPEED = 45
JUMP_FLOAT = 2
GRAVITY = 4.5 # This is in fact and for now just max fall speed
SPEED = 7
SCORE = 0
DIST_OBSTACLES = 800

class Obstacle(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__() 
        self.image = pygame.image.load("./data/candy.png")
        self.size = self.image.get_size()
        self.rect = self.image.get_rect()
        self.initial_cener = (SCREEN_WIDTH,SCREEN_HEIGHT - FLOOR_HEIGHT - self.size[1]/2)
        self.rect.center = self.initial_cener
 
    def move(self):
        global SCORE
        self.rect.move_ip(-SPEED,0)
        if (self.rect.left < -100):
            SCORE += 1
            self.kill()

    def draw(self, surface):
        surface.blit(self.image, self.rect)

class DoubleObstacle(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        # Cargar dos imágenes
        self.image1 = pygame.image.load("./data/candy.png")
        self.image2 = pygame.image.load("./data/candy.png")  # Puedes cambiar esto por otra imagen si lo deseas

        # Obtener las dimensiones de las imágenes
        width1, height1 = self.image1.get_size()
        width2, height2 = self.image2.get_size()

        # Crear una nueva superficie con el ancho total y el mayor alto de las dos imágenes
        total_width = width1 + width2
        total_height = max(height1, height2)
        self.image = pygame.Surface((total_width, total_height), pygame.SRCALPHA)

        # Dibujar las imágenes en la nueva superficie
        self.image.blit(self.image1, (0, 0))
        self.image.blit(self.image2, (width1, 0))

        # Configurar el rectángulo del sprite
        self.rect = self.image.get_rect()
        self.initial_center = (SCREEN_WIDTH, SCREEN_HEIGHT - FLOOR_HEIGHT - total_height / 2)
        self.rect.center = self.initial_center

    def move(self):
        global SCORE
        self.rect.move_ip(-SPEED, 0)
        if self.rect.left < -100:
            SCORE += 1
            self.kill()

    def draw(self, surface):
        surface.blit(self.image, self.rect)

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__() 
        self.image = pygame.image.load("./data/grinch.png")
        self.size = self.image.get_size()
        self.rect = self.image.get_rect()
        self.initial_cener = (self.size[0]/2 + 50, SCREEN_HEIGHT - FLOOR_HEIGHT - self.size[1]/2)
        self.rect.center = self.initial_cener

        self.initial_pos = (self.rect.x, self.rect.y)
        self.pos = list(self.initial_pos)
        self.vertical_speed = 0
 
    def move(self):
        pressed_keys = pygame.key.get_pressed()
        self.pos[1] = self.rect.y

        if self.pos[1] < self.initial_pos[1]:
            if self.vertical_speed < GRAVITY:
                self.vertical_speed += GRAVITY
            if pressed_keys[pygame.K_UP]:
                self.vertical_speed -= JUMP_FLOAT
            if pressed_keys[pygame.K_DOWN]:
                if self.vertical_speed < 10:
                    self.vertical_speed += GRAVITY
        else:
            self.vertical_speed = 0
            if pressed_keys[pygame.K_UP]:
                self.vertical_speed -= JUMP_SPEED

        # Evitar que el jugador caiga por debajo del suelo
        if self.pos[1] > self.initial_pos[1]:
            self.pos[1] = self.initial_pos[1]
            self.rect.y = self.initial_pos[1]
            self.vertical_speed = 0

        self.rect.move_ip(0, self.vertical_speed)

    def draw(self, surface):
        surface.blit(self.image, self.rect)

def run_game():
    # pygame setup
    pygame.init()
    global SPEED
    font = pygame.font.SysFont("Verdana", 40)

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    background_image = pygame.image.load('./data/background.png').convert()
    background_image = pygame.transform.scale(background_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

    game_over_image = pygame.image.load('./data/Game_over.png').convert()
    game_over_image = pygame.transform.scale(game_over_image, (SCREEN_WIDTH, SCREEN_HEIGHT))


    screen.blit(background_image, (0, 0))
    pygame.display.set_caption("Game")

    P1 = Player()
  

    #Creating Sprites Groups
    obstacles = pygame.sprite.Group()
   
    all_sprites = pygame.sprite.Group()
    all_sprites.add(P1)

    INC_SPEED = pygame.USEREVENT + 1
    pygame.time.set_timer(INC_SPEED, 5000)

    last_dist = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == INC_SPEED:
                if SPEED < 20:
                    SPEED += 1
        

        last_dist += SPEED
        rand = random.randint(1, 6)
        if last_dist >= DIST_OBSTACLES and (rand == 1 or rand == 2):
            new_obstacle = Obstacle()
            obstacles.add(new_obstacle)
            all_sprites.add(new_obstacle)
            last_dist = 0
        if last_dist >= DIST_OBSTACLES and rand == 3:
            new_obstacle = DoubleObstacle()
            obstacles.add(new_obstacle)
            all_sprites.add(new_obstacle)
            last_dist = 0
           

        screen.blit(background_image, (0, 0))
        scores = font.render("Score: " + str(SCORE), True, RED)
        screen.blit(scores, (10, 10))
        
        #Moves and Re-draws all Sprites
        for entity in all_sprites:
            screen.blit(entity.image, entity.rect)
            entity.move()

        # To be run if collision occurs between Player and Obstacle
        if pygame.sprite.spritecollideany(P1, obstacles):
            screen.blit(game_over_image, (0, 0))
            pygame.display.update()
            for entity in all_sprites:
                    entity.kill() 
            running = False
            time.sleep(2)


        pygame.display.update()
        clock.tick(FPS)


    pygame.quit()
    return SCORE

def main():
    final_score = run_game()
    print("Juego terminado. Puntuación final:", final_score)

if __name__ == "__main__":
    main()