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

sleep(0.5)

initial = [0 for compass in compasses]
for i, compass in enumerate(compasses):
    if compass.enabled:
        initial[i] = compass.read()
        
sleep(0.5)

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    
    screen.fill("#313131")
    
    degs = []
    for i, compass in enumerate(compasses):
        if not compass.enabled:
            pygame.draw.line(screen, colours[i % 2], (400 - 250, 300 - 250), (400 + 250, 300 + 250), 4)
            continue
            
        yaw_deg = compass.read() - initial[i]
        degs.append(yaw_deg)
        screen.blit(pygame.font.SysFont(None, 33).render(f"compass {i + 1}: {yaw_deg:.2f} deg", 1, (255, 255, 255)), (20, 20 + i * 50))
        
        yaw = math.radians(yaw_deg)
    
        dx = math.sin(yaw) * 250
        dy = math.cos(yaw) * 250
        
        pygame.draw.line(screen, colours[i % 2], (400, 300), (400 + dx, 300 + dy), 10)

    mean = degs[0] + 0.5 * ((degs[1] - degs[0] + 180) % 360 - 180)
    screen.blit(pygame.font.SysFont(None, 33).render(f"Mean: {mean:.2f} deg", 1, (255, 255, 255)), (20, 20 + 2 * 50))
        
    dx = math.sin(math.radians(mean)) * 67
    dy = math.cos(math.radians(mean)) * 67
    
    pygame.draw.line(screen, (255, 230, 100), (400, 300), (400 + dx, 300 + dy), 10)

    pygame.display.flip()
