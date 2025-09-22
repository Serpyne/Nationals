from ahrs.filters import Madgwick
import numpy as np
from time import sleep, perf_counter as now

import pygame, sys
import math
from pygame.locals import *
from compass import Compass

pygame.init()

screen = pygame.display.set_mode((800, 600))

colours = [(255, 0, 0), (0, 0, 255)]
compasses = [Compass(0x4a), Compass(0x4b)]

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    
    screen.fill("#313131")
    
    for i, compass in enumerate(compasses):
        yaw_deg = 0
        screen.blit(pygame.font.SysFont(None, 33).render(f"compass {i + 1}: {yaw_deg:2f} deg", 1, (255, 255, 255)), (20, 20 + i * 50))
        
        if not compass.enabled:
            pygame.draw.line(screen, colours[i % 2], (400 - 250, 300 - 250), (400 + 250, 300 + 250), 4)
            continue
            
        yaw_deg = compass.read()
        yaw = math.radians(yaw_deg)
    
        dx = math.sin(yaw) * 250
        dy = math.cos(yaw) * 250
        
        pygame.draw.line(screen, colours[i % 2], (400 - dx, 300 - dy), (400 + dx, 300 + dy), 10)

    pygame.display.flip()
