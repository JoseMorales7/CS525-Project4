# %%
import pygame

# %%
pygame.init()

# %%
screen = pygame.display.set_mode((1450, 650))

# Initializing RGB Color
color = (255, 0, 0)
 
# Changing surface color
screen.fill(color)
pygame.display.flip()
# screen.fill((0,0,0))


# %%
pygame.display.set_caption("TEsting")

# %%
screen.fill((255,255,255))

count = 0
while count < 1e7:
    count+=1
pygame.quit()


