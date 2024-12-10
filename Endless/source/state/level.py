__author__ = 'marble_xu'

import os
import json
import pygame as pg
import numpy as np
from torch import randint
from .. import tool
from .. import constants as c
from ..component import map, plant, zombie, menubar
import random

card1Loc = (109, 64) 
card2Loc = (153, 56) 
card3Loc = (215, 40) 
card4Loc = (267, 40) 
card5Loc = (323, 43) 
card6Loc = (372, 43) 
card7Loc = (437, 42) 
card8Loc = (494, 42) 

cardLocs = [card1Loc, card2Loc, card3Loc, card4Loc, card5Loc, card6Loc, card7Loc, card8Loc]
keys = [pg.K_1, pg.K_2, pg.K_3, pg.K_4, pg.K_5, pg.K_6, pg.K_7, pg.K_8]

# pos: (74, 143)  #1,1
# pos: (148, 142) #1,2
# pos: (226, 142) #1,3
# pos: (323, 142) #1,4
# pos: (393, 142) #1,5
# pos: (77, 237) #2,1
# pos: (144, 235) #2,2
# pos: (212, 235) #2,3
# pos: (321, 234) #2,4

gridLocs = [[((x + 1) * c.GRID_X_SIZE, (y + 1) * c.GRID_Y_SIZE + 40) for x in range(0, c.GRID_X_LEN)] for y in range(0, c.GRID_Y_LEN)]

choosePlant1Loc =  (45, 159)   # SunFlower
choosePlant2Loc =  (103, 162)  # Peashooter
choosePlant3Loc =  (206, 166)  # Walnut
choosePlant4Loc =  (99, 238)   # PotatoMine
choosePlant5Loc =  (157, 168)  # FreezeShooter
choosePlant6Loc =  (265, 166)  # Cherry bomb
choosePlant7Loc =  (363, 156)  # Double shooter
choosePlant8Loc =  (427, 165)  # Pirinha plant

choosePlantsLoc = [choosePlant1Loc, choosePlant2Loc, choosePlant3Loc, choosePlant4Loc, choosePlant5Loc, choosePlant6Loc, choosePlant7Loc, choosePlant8Loc]

Level1Zombies = [c.NORMAL_ZOMBIE, c.FLAG_ZOMBIE]
Level2Zombies = [c.NORMAL_ZOMBIE, c.FLAG_ZOMBIE, c.CONEHEAD_ZOMBIE]
Level3Zombies = [c.NORMAL_ZOMBIE, c.FLAG_ZOMBIE, c.CONEHEAD_ZOMBIE, c.BUCKETHEAD_ZOMBIE]
Level4Zombies = [c.FLAG_ZOMBIE, c.CONEHEAD_ZOMBIE, c.BUCKETHEAD_ZOMBIE]
Level5Zombies = [c.CONEHEAD_ZOMBIE, c.BUCKETHEAD_ZOMBIE]

zombiLvlLists = [Level1Zombies, Level2Zombies, Level3Zombies, Level4Zombies, Level5Zombies]

class Level(tool.State):
    def __init__(self):
        tool.State.__init__(self)
        self.numPlants = 4
        self.actionSpaceN = c.GRID_Y_LEN * c.GRID_X_LEN * self.numPlants + 2
    
    def startup(self, current_time, persist):
        self.game_info = persist
        self.persist = self.game_info
        self.game_info[c.CURRENT_TIME] = current_time
        self.map_y_len = c.GRID_Y_LEN
        self.map = map.Map(c.GRID_X_LEN, self.map_y_len)
        self.lastOutput = ""
        self.rc = 0
        self.cc = 0
        self.PlantGrid = np.zeros((c.GRID_Y_LEN, c.GRID_X_LEN + 2))
        self.ZombieGrid = np.zeros((c.GRID_Y_LEN, c.GRID_X_LEN + 2))
        self.score = 0
        self.zombieWaveTime = self.current_time
        self.zombieLvl = 0
        
        self.loadMap()
        self.setupBackground()
        self.initState()

    def loadMap(self):
        # print(f"LoadMap: {self.game_info[c.LEVEL_NUM]}")
        map_file = 'level_' + str(self.game_info[c.LEVEL_NUM]) + '.json'
        file_path = os.path.join('source', 'data', 'map', map_file)
        f = open(file_path)
        self.map_data = json.load(f)
        f.close()
    
    def setupBackground(self):
        img_index = self.map_data[c.BACKGROUND_TYPE]
        self.background_type = img_index
        self.background = tool.GFX[c.BACKGROUND_NAME][img_index]
        self.bg_rect = self.background.get_rect()

        self.level = pg.Surface((self.bg_rect.w, self.bg_rect.h)).convert()
        self.viewport = tool.SCREEN.get_rect(bottom=self.bg_rect.bottom)
        self.viewport.x += c.BACKGROUND_OFFSET_X
    
    def setupGroups(self):
        self.sun_group = pg.sprite.Group()
        self.head_group = pg.sprite.Group()

        self.plant_groups = []
        self.zombie_groups = []
        self.hypno_zombie_groups = [] #zombies who are hypno after eating hypnoshroom
        self.bullet_groups = []
        for i in range(self.map_y_len):
            self.plant_groups.append(pg.sprite.Group())
            self.zombie_groups.append(pg.sprite.Group())
            self.hypno_zombie_groups.append(pg.sprite.Group())
            self.bullet_groups.append(pg.sprite.Group())
    
    def setupZombies(self):
        def takeTime(element):
            return element[0]

        self.zombie_list = []
        zombie = random.choice(Level1Zombies)
        time = random.randint(10000, 20000)
        lane = random.randint(0,4)
        self.zombie_list.append((time, zombie, lane))
        # for data in self.map_data[c.ZOMBIE_LIST]:
        #     self.zombie_list.append((data['time'], data['name'], data['map_y']))
        self.zombie_start_time = 0
        self.zombie_list.sort(key=takeTime)

    def setupCars(self):
        self.cars = []
        for i in range(self.map_y_len):
            _, y = self.map.getMapGridPos(0, i)
            self.cars.append(plant.Car(-25, y+20, i))

    def update(self, surface, current_time, mouse_pos, mouse_click, keyPressed = None, action = None):
        self.current_time = self.game_info[c.CURRENT_TIME] = current_time
        # if self.state == c.CHOOSE:
        #     self.chooseAgent()
        if self.state == c.PLAY:
            if action is not None:
                self.playAgent(action)
            else:
                self.play(mouse_pos, mouse_click, keyPressed)
            pGrid = np.zeros((c.GRID_Y_LEN, c.GRID_X_LEN + 2))
            zGrid = np.zeros((c.GRID_Y_LEN, c.GRID_X_LEN + 2))
            for group in self.plant_groups:
                for plant in group.sprites():
                    gridX, gridY = self.map.getMapIndex(plant.rect.centerx, plant.rect.centery)
                    # print("plant")
                    # print(gridX, gridY)
                    pGrid[gridY, gridX + 1] = self.plantNameToNum(plant.name)
                    # self.score += 0.5 / c.FPS
                    
                
            for group in self.zombie_groups:
                for zombie in group.sprites():
                    zombiePos = zombie.rect #x, y, w, h
                    gridX, gridY = self.map.getMapIndex(zombiePos[0], zombiePos[1] + zombiePos[3])
                    # print("zombie")
                    # print(gridX, gridY)
                    zGrid[gridY, gridX + 1] += zombie.health

                    if zombie.health == 0 and not zombie.dead:
                        self.score += self.zomScoreWhenKills(zombie.name)
                        zombie.dead = True

            for car in self.cars:
                gridX, gridY = self.map.getMapIndex(car.rect.x, car.rect.bottom)
                # print("Car")
                # print(gridX, gridY)
                pGrid[gridY, gridX + 1] = self.plantNameToNum(c.CAR)

            self.PlantGrid = pGrid
            self.ZombieGrid = zGrid
            # self.score += 0.001 / c.FPS
            # print(self.score)

            # print(pGrid)
            # print(zGrid)
            # print(current_time)
            # print(self.menubar.getAvailableMoves()[:4])
            # print(self.menubar.sun_value)

        self.draw(surface)

    def initBowlingMap(self):
        print('initBowlingMap')
        for x in range(3, self.map.width):
            for y in range(self.map.height):
                self.map.setMapGridType(x, y, c.MAP_EXIST)

    def initState(self):
        if c.CHOOSEBAR_TYPE in self.map_data:
            self.bar_type = self.map_data[c.CHOOSEBAR_TYPE]
        else:
            self.bar_type = c.CHOOSEBAR_STATIC

        if self.bar_type == c.CHOOSEBAR_STATIC:
            self.initChoose()
        else:
            card_pool = menubar.getCardPool(self.map_data[c.CARD_POOL])
            self.initPlay(card_pool)
            if self.bar_type == c.CHOSSEBAR_BOWLING:
                self.initBowlingMap()

    def initChoose(self):
        self.state = c.CHOOSE
        self.panel = menubar.Panel(menubar.all_card_list, self.map_data[c.INIT_SUN_NAME])

        for loc in choosePlantsLoc:
            self.panel.checkCardClick(loc)

        self.initPlay(self.panel.getSelectedCards())

    def choose(self, mouse_pos, mouse_click):
        if mouse_pos and mouse_click[0]:
            self.panel.checkCardClick(mouse_pos)
            if self.panel.checkStartButtonClick(mouse_pos):
                self.initPlay(self.panel.getSelectedCards())
            
    # def chooseAgent(self):
        # if mouse_pos and mouse_click[0]:
        #     self.panel.checkCardClick(mouse_pos)
        #     if self.panel.checkStartButtonClick(mouse_pos):
        #         self.initPlay(self.panel.getSelectedCards())

        # for loc in choosePlantsLoc:
        #     self.panel.checkCardClick(loc)

        # self.initPlay(self.panel.getSelectedCards())
        # print("Done")

    def initPlay(self, card_list):
        self.state = c.PLAY
        if self.bar_type == c.CHOOSEBAR_STATIC:
            self.menubar = menubar.MenuBar(card_list, self.map_data[c.INIT_SUN_NAME])
        else:
            self.menubar = menubar.MoveBar(card_list)
        self.drag_plant = False
        self.hint_image = None
        self.hint_plant = False
        if self.background_type == c.BACKGROUND_DAY and self.bar_type == c.CHOOSEBAR_STATIC:
            self.produce_sun = True
        else:
            self.produce_sun = False
        self.sun_timer = self.current_time

        self.removeMouseImage()
        self.setupGroups()
        self.setupZombies()
        self.setupCars()

    def getActionMask(self):
        mask = np.ones(self.actionSpaceN)
        mask[0] = 1
        mask[1] = len(self.sun_group) > 0
        availableActions = np.tile(self.menubar.getAvailableMoves()[:4], c.GRID_X_LEN * c.GRID_Y_LEN)
        # print(self.menubar.getAvailableMoves().shape)
        # print(availableActions)
        grid = np.array(self.map.map).flatten().repeat(self.numPlants)
        # print(np.array(self.map.map).flatten().shape)
        # print(grid)
        mask[2:] = availableActions * np.logical_not(grid)
        return mask




    def play(self, mouse_pos, mouse_click, keyPressed):
        if self.zombie_start_time == 0:
            self.zombie_start_time = self.current_time
        elif len(self.zombie_list) > 0:
            data = self.zombie_list[0]
            if  data[0] <= (self.current_time - self.zombie_start_time):
                self.createZombie(data[1], data[2])
                self.zombie_list.remove(data)

        for i in range(self.map_y_len):
            self.bullet_groups[i].update(self.game_info)
            self.plant_groups[i].update(self.game_info)
            self.zombie_groups[i].update(self.game_info)
            self.hypno_zombie_groups[i].update(self.game_info)
            for zombie in self.hypno_zombie_groups[i]:
                if zombie.rect.x > c.SCREEN_WIDTH:
                    zombie.kill()

        self.head_group.update(self.game_info)
        self.sun_group.update(self.game_info)

        output = ""

        output += "Play before drag?\n"
        output += f"DragPlant: {self.drag_plant}\n"
        output += f"Mouse Pos: {mouse_pos}\n"
        output += f"Mouse Click: {mouse_click[0]}\n"

        # def checkCardClick(self, mouse_pos):
        # result = None
        # for card in self.card_list:
        #     if card.checkMouseClick(mouse_pos):
        #         if card.canClick(self.sun_value, self.current_time):
        #             result = (plant_name_list[card.name_index], card)
        #         break

        # if mouse_click[0]:
        #     #collect sun
        #     # if len(self.sun_group) > 0:
        #     #     sun = list(self.sun_group)[0]
        #     #     sun.checkCollision(sun.rect.x, sun.rect.y)
        #     #     self.menubar.increaseSunValue(sun.sun_value)

        #     # print(menubar.plant_name_list[self.menubar.card_list[0]])
        #     # print(type(menubar.plant_name_list[self.menubar.card_list[0]]))
        #     # print(self.menubar.card_list[0])
        #     # print(type(self.menubar.card_list[0]))
        #     self.setupMouseImage(menubar.plant_name_list[self.menubar.card_list[0].name_index], self.menubar.card_list[0])
        #     self.addPlantAgent()
        #     self.removeMouseImage()

        if keyPressed[pg.K_q]:
            #collect sun
            if len(self.sun_group) > 0:
                sun = list(self.sun_group)[0]
                sun.checkCollision(sun.rect.x, sun.rect.y)
                self.menubar.increaseSunValue(sun.sun_value)

                self.score += 0.25



        for key, cardLoc in zip(keys, cardLocs):
            if keyPressed[key]:
                result = self.menubar.checkCardClick(cardLoc)
                output += f"{result}\n"
                if result:
                    output += "setupMouseImage\n"
                    self.setupMouseImage(result[0], result[1])

                    self.addPlantAgent(gridLocs[self.rc][self.cc])
                    self.removeMouseImage()
                    self.cc += 1

                    if self.cc == c.GRID_X_LEN:
                        self.rc += 1
                        self.cc = 0

                break  # Exit loop after processing one key

        if not self.drag_plant and mouse_pos and mouse_click[0]:
            output += "First if"
            result = self.menubar.checkCardClick(mouse_pos)
            output += f"{result}\n"
            if result:
                output += "setupMouseImage\n"
                self.setupMouseImage(result[0], result[1])
        elif self.drag_plant:
            output += "Second if dragging\n"
            if mouse_click[1]:
                output += "Removing image\n"
                self.removeMouseImage()
            elif mouse_click[0]:
                output += "Left click\n"
                # output += f"Check menu bar click: {self.menubar.checkMenuBarClick(mouse_pos)}\n"
                if self.menubar.checkMenuBarClick(mouse_pos):
                    output+="removing image\n"
                    self.removeMouseImage()
                else:
                    output += "adding plant\n"
                    self.addPlant()
            elif mouse_pos is None:
                self.setupHintImage()

        if self.lastOutput != output: 
            # print(output)
            self.lastOutput = output
        
        if self.produce_sun:
            if(self.current_time - self.sun_timer) > c.PRODUCE_SUN_INTERVAL:
                self.sun_timer = self.current_time
                map_x, map_y = self.map.getRandomMapIndex()
                x, y = self.map.getMapGridPos(map_x, map_y)
                self.sun_group.add(plant.Sun(x, 0, x, y))
        if not self.drag_plant and mouse_pos and mouse_click[0]:
            for sun in self.sun_group:
                if sun.checkCollision(mouse_pos[0], mouse_pos[1]):
                    self.menubar.increaseSunValue(sun.sun_value)

        for car in self.cars:
            car.update(self.game_info)

        self.menubar.update(self.current_time)

        self.checkBulletCollisions()
        self.checkZombieCollisions()
        self.checkPlants()
        self.checkCarCollisions()
        self.checkGameState()

    def plantNameToNum(self, name):
        if name == c.SUNFLOWER:
            return 1
        elif name == c.PEASHOOTER:
            return 2
        elif name == c.WALLNUT:
            return 3
        elif name == c.POTATOMINE:
            return 4
        elif name == c.SNOWPEASHOOTER:
            name == 5
        elif name == c.CHERRYBOMB:
            name == 6
        elif name == c.REPEATERPEA:
            name == 7
        elif name == c.CHOMPER:
            name == 8
        elif name == c.CAR:
            return 9
        else:
            print(name)
            raise NotImplementedError("No plant for this yet")
        
    def zomScoreWhenKills(self, name):
        if name == c.NORMAL_ZOMBIE:
            return c.NORMAL_HEALTH
        elif name == c.FLAG_ZOMBIE:
            return c.FLAG_HEALTH
        elif name == c.CONEHEAD_ZOMBIE:
            return c.CONEHEAD_HEALTH
        elif name == c.BUCKETHEAD_ZOMBIE:
            return c.BUCKETHEAD_HEALTH
        elif name == c.NEWSPAPER_ZOMBIE:
            return c.NEWSPAPER_HEALTH
        else:
            print(name)
            raise NotImplementedError("No Zombie for this yet")

    def addPlantAgent(self, loc):
        x, y = loc
        pos = self.map.showPlant(x, y)
        if pos is None:
            return

        if self.hint_image is None:
            self.setupHintImageAgent((x,y))
        x, y = self.hint_rect.centerx, self.hint_rect.bottom
        # print(f"x: {x}, y: {y}")
        map_x, map_y = self.map.getMapIndex(x, y)
        # print(f"mx: {map_x}, my: {map_y}")

        if self.plant_name == c.SUNFLOWER:
            new_plant = plant.SunFlower(x, y, self.sun_group, self.current_time)
            self.score += 3
        elif self.plant_name == c.PEASHOOTER:
            new_plant = plant.PeaShooter(x, y, self.bullet_groups[map_y], self.current_time)
            self.score += 3
        elif self.plant_name == c.SNOWPEASHOOTER:
            new_plant = plant.SnowPeaShooter(x, y, self.bullet_groups[map_y])
        elif self.plant_name == c.WALLNUT:
            new_plant = plant.WallNut(x, y, self.current_time)
            self.score += 3
            
        elif self.plant_name == c.CHERRYBOMB:
            new_plant = plant.CherryBomb(x, y)
        elif self.plant_name == c.THREEPEASHOOTER:
            new_plant = plant.ThreePeaShooter(x, y, self.bullet_groups, map_y)
        elif self.plant_name == c.REPEATERPEA:
            new_plant = plant.RepeaterPea(x, y, self.bullet_groups[map_y])
        elif self.plant_name == c.CHOMPER:
            new_plant = plant.Chomper(x, y)
        elif self.plant_name == c.PUFFSHROOM:
            new_plant = plant.PuffShroom(x, y, self.bullet_groups[map_y])
        elif self.plant_name == c.POTATOMINE:
            new_plant = plant.PotatoMine(x, y, self.current_time)
            self.score += 3
        elif self.plant_name == c.SQUASH:
            new_plant = plant.Squash(x, y)
        elif self.plant_name == c.SPIKEWEED:
            new_plant = plant.Spikeweed(x, y)
        elif self.plant_name == c.JALAPENO:
            new_plant = plant.Jalapeno(x, y)
        elif self.plant_name == c.SCAREDYSHROOM:
            new_plant = plant.ScaredyShroom(x, y, self.bullet_groups[map_y])
        elif self.plant_name == c.SUNSHROOM:
            new_plant = plant.SunShroom(x, y, self.sun_group)
        elif self.plant_name == c.ICESHROOM:
            new_plant = plant.IceShroom(x, y)
        elif self.plant_name == c.HYPNOSHROOM:
            new_plant = plant.HypnoShroom(x, y)
        elif self.plant_name == c.WALLNUTBOWLING:
            new_plant = plant.WallNutBowling(x, y, map_y, self)
        elif self.plant_name == c.REDWALLNUTBOWLING:
            new_plant = plant.RedWallNutBowling(x, y)

        if new_plant.can_sleep and self.background_type == c.BACKGROUND_DAY:
            new_plant.setSleep()
        self.plant_groups[map_y].add(new_plant)
        if self.bar_type == c.CHOOSEBAR_STATIC:
            self.menubar.decreaseSunValue(self.select_plant.sun_cost)
            self.menubar.setCardFrozenTime(self.plant_name)
        else:
            self.menubar.deleateCard(self.select_plant)

        # print(self.bar_type)
        if self.bar_type != c.CHOSSEBAR_BOWLING:
            # print("Setting grid type")
            self.map.setMapGridType(map_x, map_y, c.MAP_EXIST)
        self.removeMouseImage()

    def playAgent(self, action):
        if self.zombie_start_time == 0:
            self.zombie_start_time = self.current_time
        elif len(self.zombie_list) > 0:
            #(data['time'], data['name'], data['map_y'])
            data = self.zombie_list[0]
            if  data[0] <= (self.current_time - self.zombie_start_time):
                self.createZombie(data[1], data[2])
                self.zombie_list.remove(data)
            
            if len(self.zombie_list) == 0:
                zombie = random.choice(zombiLvlLists[self.zombieLvl])
                time = (self.current_time - self.zombie_start_time) + random.randint(15000 // (self.zombieLvl + 1), 30000 // (self.zombieLvl + 1))
                lane = random.randint(0,4)
                self.zombie_list.append((time, zombie, lane))

                if (self.current_time - self.zombie_start_time) - self.zombieWaveTime > 120000 and self.zombieLvl < len(zombiLvlLists) - 1:
                    self.zombieWaveTime = self.current_time - self.zombie_start_time
                    self.zombieLvl+=1

            

        for i in range(self.map_y_len):
            self.bullet_groups[i].update(self.game_info)
            self.plant_groups[i].update(self.game_info)
            self.zombie_groups[i].update(self.game_info)
            self.hypno_zombie_groups[i].update(self.game_info)
            for zombie in self.hypno_zombie_groups[i]:
                if zombie.rect.x > c.SCREEN_WIDTH:
                    zombie.kill()

        self.head_group.update(self.game_info)
        self.sun_group.update(self.game_info)
        
        if action == 1:
            #collect sun
            if len(self.sun_group) > 0:
                sun = list(self.sun_group)[0]
                sun.checkCollision(sun.rect.x, sun.rect.y)
                self.menubar.increaseSunValue(sun.sun_value)

                self.score += 0.25

        action -= 2
        row = action // (c.GRID_X_LEN * self.numPlants)
        col = (action % (c.GRID_X_LEN * self.numPlants)) // self.numPlants
        action = action % self.numPlants

        result = self.menubar.checkCardClick(cardLocs[action])
        if result:
            self.setupMouseImage(result[0], result[1])
            # print(len(gridLocs))
            # print(len(gridLocs[0]))
            # print(row)
            # print(col)
            self.addPlantAgent(gridLocs[row][col])
            self.removeMouseImage()

        
        
        if self.produce_sun:
            if(self.current_time - self.sun_timer) > c.PRODUCE_SUN_INTERVAL:
                self.sun_timer = self.current_time
                map_x, map_y = self.map.getRandomMapIndex()
                x, y = self.map.getMapGridPos(map_x, map_y)
                self.sun_group.add(plant.Sun(x, 0, x, y))
        if action == 0: #some other action that signifies to get the sent
            if len(self.sun_group) > 0:
                sun = list(self.sun_group)[0]
                sun.checkCollision(sun.rect.x, sun.rect.y)
                self.menubar.increaseSunValue(sun.sun_value)

        for car in self.cars:
            car.update(self.game_info)

        self.menubar.update(self.current_time)

        self.checkBulletCollisions()
        self.checkZombieCollisions()
        self.checkPlants()
        self.checkCarCollisions()
        self.checkGameState()

    def createZombie(self, name, map_y):
        x, y = self.map.getMapGridPos(0, map_y)
        if name == c.NORMAL_ZOMBIE:
            self.zombie_groups[map_y].add(zombie.NormalZombie(c.ZOMBIE_START_X, y, self.head_group))
        elif name == c.CONEHEAD_ZOMBIE:
            self.zombie_groups[map_y].add(zombie.ConeHeadZombie(c.ZOMBIE_START_X, y, self.head_group))
        elif name == c.BUCKETHEAD_ZOMBIE:
            self.zombie_groups[map_y].add(zombie.BucketHeadZombie(c.ZOMBIE_START_X, y, self.head_group))
        elif name == c.FLAG_ZOMBIE:
            self.zombie_groups[map_y].add(zombie.FlagZombie(c.ZOMBIE_START_X, y, self.head_group))
        elif name == c.NEWSPAPER_ZOMBIE:
            self.zombie_groups[map_y].add(zombie.NewspaperZombie(c.ZOMBIE_START_X, y, self.head_group))

    def canSeedPlant(self):
        x, y = pg.mouse.get_pos()
        # x, y = (232, 346)
        return self.map.showPlant(x, y)
    
    def canSeedPlantAgent(self, location):
        x, y = location
        return self.map.showPlant(x, y)
        
    def addPlant(self):
        pos = self.canSeedPlant()
        if pos is None:
            return

        if self.hint_image is None:
            self.setupHintImage()
        x, y = self.hint_rect.centerx, self.hint_rect.bottom
        map_x, map_y = self.map.getMapIndex(x, y)
        if self.plant_name == c.SUNFLOWER:
            new_plant = plant.SunFlower(x, y, self.sun_group)
        elif self.plant_name == c.PEASHOOTER:
            new_plant = plant.PeaShooter(x, y, self.bullet_groups[map_y])
        elif self.plant_name == c.SNOWPEASHOOTER:
            new_plant = plant.SnowPeaShooter(x, y, self.bullet_groups[map_y])
        elif self.plant_name == c.WALLNUT:
            new_plant = plant.WallNut(x, y)
        elif self.plant_name == c.CHERRYBOMB:
            new_plant = plant.CherryBomb(x, y)
        elif self.plant_name == c.THREEPEASHOOTER:
            new_plant = plant.ThreePeaShooter(x, y, self.bullet_groups, map_y)
        elif self.plant_name == c.REPEATERPEA:
            new_plant = plant.RepeaterPea(x, y, self.bullet_groups[map_y])
        elif self.plant_name == c.CHOMPER:
            new_plant = plant.Chomper(x, y)
        elif self.plant_name == c.PUFFSHROOM:
            new_plant = plant.PuffShroom(x, y, self.bullet_groups[map_y])
        elif self.plant_name == c.POTATOMINE:
            new_plant = plant.PotatoMine(x, y)
        elif self.plant_name == c.SQUASH:
            new_plant = plant.Squash(x, y)
        elif self.plant_name == c.SPIKEWEED:
            new_plant = plant.Spikeweed(x, y)
        elif self.plant_name == c.JALAPENO:
            new_plant = plant.Jalapeno(x, y)
        elif self.plant_name == c.SCAREDYSHROOM:
            new_plant = plant.ScaredyShroom(x, y, self.bullet_groups[map_y])
        elif self.plant_name == c.SUNSHROOM:
            new_plant = plant.SunShroom(x, y, self.sun_group)
        elif self.plant_name == c.ICESHROOM:
            new_plant = plant.IceShroom(x, y)
        elif self.plant_name == c.HYPNOSHROOM:
            new_plant = plant.HypnoShroom(x, y)
        elif self.plant_name == c.WALLNUTBOWLING:
            new_plant = plant.WallNutBowling(x, y, map_y, self)
        elif self.plant_name == c.REDWALLNUTBOWLING:
            new_plant = plant.RedWallNutBowling(x, y)

        if new_plant.can_sleep and self.background_type == c.BACKGROUND_DAY:
            new_plant.setSleep()
        self.plant_groups[map_y].add(new_plant)
        if self.bar_type == c.CHOOSEBAR_STATIC:
            self.menubar.decreaseSunValue(self.select_plant.sun_cost)
            self.menubar.setCardFrozenTime(self.plant_name)
        else:
            self.menubar.deleateCard(self.select_plant)

        if self.bar_type != c.CHOSSEBAR_BOWLING:
            self.map.setMapGridType(map_x, map_y, c.MAP_EXIST)
        self.removeMouseImage()
        #print('addPlant map[%d,%d], grid pos[%d, %d] pos[%d, %d]' % (map_x, map_y, x, y, pos[0], pos[1]))

    def setupHintImageAgent(self, location):
        pos = self.canSeedPlantAgent(location)
        # print(pos)
        # print(self.mouse_image)
        if pos and self.mouse_image:
            if (self.hint_image and pos[0] == self.hint_rect.x and
                pos[1] == self.hint_rect.y):
                return
            width, height = self.mouse_rect.w, self.mouse_rect.h
            image = pg.Surface([width, height])
            image.blit(self.mouse_image, (0, 0), (0, 0, width, height))
            image.set_colorkey(c.BLACK)
            image.set_alpha(128)
            self.hint_image = image
            self.hint_rect = image.get_rect()
            self.hint_rect.centerx = pos[0]
            self.hint_rect.bottom = pos[1]
            self.hint_plant = True
        else:
            self.hint_plant = False
    
    def setupHintImage(self):
        pos = self.canSeedPlant()
        # print(pos)
        # print(self.mouse_image)
        if pos and self.mouse_image:
            if (self.hint_image and pos[0] == self.hint_rect.x and
                pos[1] == self.hint_rect.y):
                return
            width, height = self.mouse_rect.w, self.mouse_rect.h
            image = pg.Surface([width, height])
            image.blit(self.mouse_image, (0, 0), (0, 0, width, height))
            image.set_colorkey(c.BLACK)
            image.set_alpha(128)
            self.hint_image = image
            self.hint_rect = image.get_rect()
            self.hint_rect.centerx = pos[0]
            self.hint_rect.bottom = pos[1]
            self.hint_plant = True
        else:
            self.hint_plant = False

    def setupMouseImage(self, plant_name, select_plant):
        frame_list = tool.GFX[plant_name]
        if plant_name in tool.PLANT_RECT:
            data = tool.PLANT_RECT[plant_name]
            x, y, width, height = data['x'], data['y'], data['width'], data['height']
        else:
            x, y = 0, 0
            rect = frame_list[0].get_rect()
            width, height = rect.w, rect.h

        if (plant_name == c.POTATOMINE or plant_name == c.SQUASH or
            plant_name == c.SPIKEWEED or plant_name == c.JALAPENO or
            plant_name == c.SCAREDYSHROOM or plant_name == c.SUNSHROOM or
            plant_name == c.ICESHROOM or plant_name == c.HYPNOSHROOM or
            plant_name == c.WALLNUTBOWLING or plant_name == c.REDWALLNUTBOWLING):
            color = c.WHITE
        else:
            color = c.BLACK
        self.mouse_image = tool.get_image(frame_list[0], x, y, width, height, color, 1)
        self.mouse_rect = self.mouse_image.get_rect()
        pg.mouse.set_visible(False)
        self.drag_plant = True
        self.plant_name = plant_name
        self.select_plant = select_plant

    def removeMouseImage(self):
        pg.mouse.set_visible(True)
        self.drag_plant = False
        self.mouse_image = None
        self.hint_image = None
        self.hint_plant = False

    def checkBulletCollisions(self):
        collided_func = pg.sprite.collide_circle_ratio(0.7)
        for i in range(self.map_y_len):
            for bullet in self.bullet_groups[i]:
                if bullet.state == c.FLY:
                    zombie = pg.sprite.spritecollideany(bullet, self.zombie_groups[i], collided_func)
                    if zombie and zombie.state != c.DIE:
                        zombie.setDamage(bullet.damage, bullet.ice)
                        bullet.setExplode()
    
    def checkZombieCollisions(self):
        if self.bar_type == c.CHOSSEBAR_BOWLING:
            ratio = 0.6
        else:
            ratio = 0.7
        collided_func = pg.sprite.collide_circle_ratio(ratio)
        for i in range(self.map_y_len):
            hypo_zombies = []
            for zombie in self.zombie_groups[i]:
                if zombie.state != c.WALK:
                    continue
                plant = pg.sprite.spritecollideany(zombie, self.plant_groups[i], collided_func)
                if plant:
                    if plant.name == c.WALLNUTBOWLING:
                        if plant.canHit(i):
                            zombie.setDamage(c.WALLNUT_BOWLING_DAMAGE)
                            plant.changeDirection(i)
                    elif plant.name == c.REDWALLNUTBOWLING:
                        if plant.state == c.IDLE:
                            plant.setAttack()
                    elif plant.name != c.SPIKEWEED:
                        zombie.setAttack(plant)

            for hypno_zombie in self.hypno_zombie_groups[i]:
                if hypno_zombie.health <= 0:
                    continue
                zombie_list = pg.sprite.spritecollide(hypno_zombie,
                               self.zombie_groups[i], False,collided_func)
                for zombie in zombie_list:
                    if zombie.state == c.DIE:
                        continue
                    if zombie.state == c.WALK:
                        zombie.setAttack(hypno_zombie, False)
                    if hypno_zombie.state == c.WALK:
                        hypno_zombie.setAttack(zombie, False)

    def checkCarCollisions(self):
        collided_func = pg.sprite.collide_circle_ratio(0.8)
        for car in self.cars:
            beginState = car.state
            zombies = pg.sprite.spritecollide(car, self.zombie_groups[car.map_y], False, collided_func)
            for zombie in zombies:
                if zombie and zombie.state != c.DIE:
                    car.setWalk()
                    zombie.setDie()
            if car.dead:
                self.cars.remove(car)

            if beginState != car.state:
                # print("Starting mower")
                self.score -= 10



    def boomZombies(self, x, map_y, y_range, x_range):
        for i in range(self.map_y_len):
            if abs(i - map_y) > y_range:
                continue
            for zombie in self.zombie_groups[i]:
                if abs(zombie.rect.centerx - x) <= x_range:
                    zombie.setBoomDie()

    def freezeZombies(self, plant):
        for i in range(self.map_y_len):
            for zombie in self.zombie_groups[i]:
                if zombie.rect.centerx < c.SCREEN_WIDTH:
                    zombie.setFreeze(plant.trap_frames[0])

    def killPlant(self, plant):
        x, y = plant.getPosition()
        map_x, map_y = self.map.getMapIndex(x, y)
        if self.bar_type != c.CHOSSEBAR_BOWLING:
            self.map.setMapGridType(map_x, map_y, c.MAP_EMPTY)
        if (plant.name == c.CHERRYBOMB or plant.name == c.JALAPENO or
            (plant.name == c.POTATOMINE and not plant.is_init) or
            plant.name == c.REDWALLNUTBOWLING):
            self.boomZombies(plant.rect.centerx, map_y, plant.explode_y_range,
                            plant.explode_x_range)
        elif plant.name == c.ICESHROOM and plant.state != c.SLEEP:
            self.freezeZombies(plant)
        elif plant.name == c.HYPNOSHROOM and plant.state != c.SLEEP:
            zombie = plant.kill_zombie
            zombie.setHypno()
            _, map_y = self.map.getMapIndex(zombie.rect.x, zombie.rect.bottom)
            self.zombie_groups[map_y].remove(zombie)
            self.hypno_zombie_groups[map_y].add(zombie)
        plant.kill()

    def checkPlant(self, plant, i):
        zombie_len = len(self.zombie_groups[i])
        if plant.name == c.THREEPEASHOOTER:
            if plant.state == c.IDLE:
                if zombie_len > 0:
                    plant.setAttack()
                elif (i-1) >= 0 and len(self.zombie_groups[i-1]) > 0:
                    plant.setAttack()
                elif (i+1) < self.map_y_len and len(self.zombie_groups[i+1]) > 0:
                    plant.setAttack()
            elif plant.state == c.ATTACK:
                if zombie_len > 0:
                    pass
                elif (i-1) >= 0 and len(self.zombie_groups[i-1]) > 0:
                    pass
                elif (i+1) < self.map_y_len and len(self.zombie_groups[i+1]) > 0:
                    pass
                else:
                    plant.setIdle()
        elif plant.name == c.CHOMPER:
            for zombie in self.zombie_groups[i]:
                if plant.canAttack(zombie):
                    plant.setAttack(zombie, self.zombie_groups[i])
                    break
        elif plant.name == c.POTATOMINE:
            for zombie in self.zombie_groups[i]:
                if plant.canAttack(zombie):
                    plant.setAttack()
                    break
        elif plant.name == c.SQUASH:
            for zombie in self.zombie_groups[i]:
                if plant.canAttack(zombie):
                    plant.setAttack(zombie, self.zombie_groups[i])
                    break
        elif plant.name == c.SPIKEWEED:
            can_attack = False
            for zombie in self.zombie_groups[i]:
                if plant.canAttack(zombie):
                    can_attack = True
                    break
            if plant.state == c.IDLE and can_attack:
                plant.setAttack(self.zombie_groups[i])
            elif plant.state == c.ATTACK and not can_attack:
                plant.setIdle()
        elif plant.name == c.SCAREDYSHROOM:
            need_cry = False
            can_attack = False
            for zombie in self.zombie_groups[i]:
                if plant.needCry(zombie):
                    need_cry = True
                    break
                elif plant.canAttack(zombie):
                    can_attack = True
            if need_cry:
                if plant.state != c.CRY:
                    plant.setCry()
            elif can_attack:
                if plant.state != c.ATTACK:
                    plant.setAttack()
            elif plant.state != c.IDLE:
                plant.setIdle()
        elif(plant.name == c.WALLNUTBOWLING or
             plant.name == c.REDWALLNUTBOWLING):
            pass
        else:
            can_attack = False
            if (plant.state == c.IDLE and zombie_len > 0):
                for zombie in self.zombie_groups[i]:
                    if plant.canAttack(zombie):
                        can_attack = True
                        break
            if plant.state == c.IDLE and can_attack:
                plant.setAttack()
            elif (plant.state == c.ATTACK and not can_attack):
                plant.setIdle()

    def checkPlants(self):
        for i in range(self.map_y_len):
            for plant in self.plant_groups[i]:
                if plant.state != c.SLEEP:
                    self.checkPlant(plant, i)
                if plant.health <= 0:
                    self.killPlant(plant)

    def checkVictory(self):
        if len(self.zombie_list) > 0:
            return False
        for i in range(self.map_y_len):
            if len(self.zombie_groups[i]) > 0:
                return False
        return True
    
    def checkLose(self):
        for i in range(self.map_y_len):
            for zombie in self.zombie_groups[i]:
                if zombie.rect.right < 0:
                    return True
        return False

    def checkGameState(self):
        if self.checkVictory():
            # print(f"checkGameState: {self.game_info[c.LEVEL_NUM]}")
            self.game_info[c.LEVEL_NUM] = 1 #random.randint(1,5)
            self.next = c.LEVEL
            self.done = True
            self.score += 10
        elif self.checkLose():
            # print(f"checkGameState: {self.game_info[c.LEVEL_NUM]}")
            self.game_info[c.LEVEL_NUM] = 1 #random.randint(1,5)
            self.next = c.LEVEL
            self.done = True
            self.score -= 10

    def drawMouseShow(self, surface):
        if self.hint_plant:
            surface.blit(self.hint_image, self.hint_rect)
        x, y = pg.mouse.get_pos()
        self.mouse_rect.centerx = x
        self.mouse_rect.centery = y
        surface.blit(self.mouse_image, self.mouse_rect)

    def drawZombieFreezeTrap(self, i, surface):
        for zombie in self.zombie_groups[i]:
            zombie.drawFreezeTrap(surface)

    def draw(self, surface):
        self.level.blit(self.background, self.viewport, self.viewport)
        surface.blit(self.level, (0,0), self.viewport)
        if self.state == c.CHOOSE:
            self.panel.draw(surface)
        elif self.state == c.PLAY:
            self.menubar.draw(surface)
            for i in range(self.map_y_len):
                self.plant_groups[i].draw(surface)
                self.zombie_groups[i].draw(surface)
                self.hypno_zombie_groups[i].draw(surface)
                self.bullet_groups[i].draw(surface)
                self.drawZombieFreezeTrap(i, surface)
            for car in self.cars:
                car.draw(surface)
            self.head_group.draw(surface)
            self.sun_group.draw(surface)

            if self.drag_plant:
                self.drawMouseShow(surface)

    def getScore(self):
        return self.score