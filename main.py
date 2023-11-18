import pygame
import os
import sys
import math
import neat

pygame.init()

CLOCK = pygame.time.Clock()
FPS = 60

SCREEN_WIDTH = 1244*0.8
SCREEN_HEIGHT = 1016*0.8
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("NEAT Demo")

white = (255,255,255)
font = pygame.font.SysFont("Arial", 40)

TRACK = pygame.transform.scale(pygame.image.load("Assets/track.png").convert_alpha(),(SCREEN_WIDTH, SCREEN_HEIGHT))

class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.car_img = pygame.image.load("Assets/car.png").convert_alpha()
        self.original_image =pygame.transform.scale(self.car_img, (self.car_img.get_width(), self.car_img.get_height()))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(375, 655))
        self.drive_state = True
        self.vel_vector = pygame.math.Vector2(0.7, 0)
        self.angle = 0
        self.rotation_vel = 2
        self.direction = 0
        self.alive = True
        self.radars = []


    def update(self):
        self.radars.clear()
        self.drive()
        self. rotate()
        #Radar雷达
        for radar_angle in (-60, -30, 0, 30, 60):
            self.radar(radar_angle)
        self.collision()
        self.data()

    def drive(self):
        if self.drive_state:
           self.rect.center += self.vel_vector * 3

    def rotate(self):
        if self.direction == 1:
           self.angle -= self.rotation_vel
           self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
           self.angle += self.rotation_vel
           self.vel_vector.rotate_ip(-self.rotation_vel)

           self.image = pygame.transform.rotozoom(self.original_image, self.angle, 1)
           self.rect = self.image.get_rect(center=self.rect.center)

    def radar(self, radar_angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])
        while not SCREEN.get_at((x, y)) == pygame.Color(42,99,41,255) and length < 120: # 获取像素点颜色 get the color
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians (self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)
        # 画雷达
        pygame.draw.line(SCREEN, (255, 255, 255, 255), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (0, 255, 0, 0), (x, y), 3)
        dist = int(math.sqrt(math.pow(self.rect.center[0] - x, 2) + math.pow(self.rect.center[1] - y, 2)))
        self.radars.append([radar_angle, dist])

    def collision(self):
        length = 40
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
                                 int (self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians (self.angle - 18)) * length),
                                int(self.rect.center[1] - math.sin(math.radians (self.angle - 18)) * length)]
        # 检测碰搅 detect collision
        if SCREEN.get_at(collision_point_right) == pygame.Color(42, 99, 41, 255) \
            or SCREEN. get_at (collision_point_left) == pygame.Color(42, 99, 41, 255):
            self.alive = False
            pygame.draw.circle(SCREEN, (0, 255, 0, 50), collision_point_right, 15)
            pygame.draw.circle(SCREEN, (0, 255, 0, 50), collision_point_left, 15)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_right, 5)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_left, 5)

    def data(self):
        input = [0, 0, 0, 0, 0]
        for i, radar in enumerate(self.radars):

            input[i] = int(radar[1])

        return input

car = pygame.sprite.GroupSingle(Car())

#画文本
def draw_text(text, font, text_color, x, y):
    img = font.render(text, True, text_color)
    SCREEN.blit(img,(x,y))

def eval_genomes (genomes, config):
    global cars, ge, nets
    cars = []
    ge = []
    nets = []
    for genome_id, genome in genomes:
        cars.append(pygame.sprite.GroupSingle(Car()))
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
               pygame.quit()
               sys.exit()
        SCREEN.blit(TRACK, (0, 0))

        ## 用户输入
        # user_input = рудате.key.get_pressed()
        # if sum(pygame.key.get_pressed()) ‹= 1:
        #    car.sprite.drive_state = False
        #
        ##驾驶
        # if user_input[pygame.K_UP]:
        #
        #    car.sprite.drive_state = True
        #
        #    car.sprite.direction = 0
        #
        ## 旋转
        # if user_input[pygame.K_RIGHT]:
        #
        #    car.sprite.direction = 1
        #
        # if user_input[pygame.K_RIGHT]:
        #    car.sprite.direction = -1

        if len(cars) == 0:
           return
        for i, car in enumerate(cars):
            if not car.sprite.alive:
                ge[i].fitness -= 1
                cars.pop(i)
                ge.pop(i)
                nets.pop(i)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.sprite.data())
            if output[0] > 0.7:
                car.sprite.direction = 1
            if output[1] > 0.7:
                car.sprite.direction = -1
            if output[0] <= 0.7 and output[1] <=0.7:
                car.sprite.direction = 0



        #更新
        for car in cars:

            car.draw(SCREEN)
            car.update()

        draw_text(f"Generation: {population.generation} alive numbers: {len(cars)}", font, white, SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2)
        CLOCK.tick(FPS)
        pygame.display.flip()


config = neat.config.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "config.txt"
)
population = neat.Population(config)
stats = neat.StatisticsReporter()
population.add_reporter(stats)
population.run(eval_genomes, 500)



# if __name__ == '__main__':
#     eval_genomes()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
