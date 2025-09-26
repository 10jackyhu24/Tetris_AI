import pygame
import random
from AI_DQN import TetrisAI
from action_decoder import action_decoder
import torch
import numpy


class Window:
    def __init__(self, win_size: tuple, title: str) -> None:
        self.screen = pygame.display.set_mode(win_size)
        pygame.display.set_caption(title)
        self.events = {
            'quit': False,
            'left_key_down': False,
            'timer': 0,
            'Z_down': False,
            'X_down': False,
            'C_down': False,
            'Left_down': False,
            'Down_down': False,
            'Right_down': False,
            'space_down': False
        }

    def is_mouse_on_obj(self, boj_pos: tuple, boj_size: tuple) -> bool:
        mouse_pos = pygame.mouse.get_pos()
        if boj_pos[0] <= mouse_pos[0] <= boj_pos[0] + boj_size[0] and boj_pos[1] <= mouse_pos[1] <= boj_pos[1] + boj_size[1]:
            return True
        else:
            return False
        
    def event_detect(self, play_mode = 0) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.events['quit'] = True

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pressed()[0]:
                    self.events['left_key_down'] = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self.events['left_key_down'] = False

            if play_mode == 1:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_z:
                        self.events['Z_down'] = True
                    elif event.key == pygame.K_x:
                        self.events['X_down'] = True
                    elif event.key == pygame.K_c:
                        self.events['C_down'] = True
                    elif event.key == pygame.K_LEFT:
                        self.events['Left_down'] = True
                    elif event.key == pygame.K_DOWN:
                        self.events['Down_down'] = True
                    elif event.key == pygame.K_RIGHT:
                        self.events['Right_down'] = True
                    elif event.key == pygame.K_SPACE:
                        self.events['space_down'] = True

                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_z:
                        self.events['Z_down'] = False
                    elif event.key == pygame.K_x:
                        self.events['X_down'] = False
                    elif event.key == pygame.K_c:
                        self.events['C_down'] = False
                    elif event.key == pygame.K_LEFT:
                        self.events['Left_down'] = False
                    elif event.key == pygame.K_DOWN:
                        self.events['Down_down'] = False
                    elif event.key == pygame.K_RIGHT:
                        self.events['Right_down'] = False
                    elif event.key == pygame.K_SPACE:
                        self.events['space_down'] = False

            if event.type == pygame.USEREVENT:
                self.events['timer'] += 1
        
    def draw(self, color: tuple, pos: tuple, size: tuple) -> None:
        pygame.draw.rect(self.screen, color, (pos, size))

    def button(self, color1: tuple, color2: tuple, pos: tuple, size: tuple) -> bool:
        if self.is_mouse_on_obj(pos, size):
            pygame.draw.rect(self.screen, color2, (pos, size))
            if self.events['left_key_down']:
                return True
        else:
            pygame.draw.rect(self.screen, color1, (pos, size))
        return False

    def draw_text(self, text: str, font_size: int, text_color: tuple, boj_pos: tuple, boj_size: tuple) -> None:
        font = pygame.font.Font(None, font_size)
        text_surface = font.render(text, True, text_color)
        text_rect = text_surface.get_rect()
        text_rect.center = (boj_pos[0] + boj_size[0] // 2, boj_pos[1] + boj_size[1] // 2)
        self.screen.blit(text_surface, text_rect)


class Block:
    def __init__(self) -> None:
        self.BLOCKS = {
            'I': [[0, 0, 0, 0],
                  [1, 1, 1, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]],
            'J': [[2, 0, 0],
                  [2, 2, 2],
                  [0, 0, 0]],
            'L': [[0, 0, 3],
                  [3, 3, 3],
                  [0, 0, 0]],
            'O': [[4, 4],
                  [4, 4]],
            'S': [[0, 5, 5],
                  [5, 5, 0],
                  [0, 0, 0]],
            'T': [[0, 6, 0],
                  [6, 6, 6],
                  [0, 0, 0]],
            'Z': [[7, 7, 0],
                  [0, 7, 7],
                  [0, 0, 0]]
        }

        self.type = None
        self.block_matrix = None
        self.position = None
        self.rotation = 0
        self.touched_ground = False

    def clone(self) -> 'Block':
        new_block = Block()
        new_block.type = self.type
        new_block.block_matrix = [row[:] for row in self.block_matrix]
        new_block.position = self.position[:]
        new_block.rotation = self.rotation
        return new_block
    
    def get_block_matrix(self, block_type: str):
        return self.BLOCKS[block_type]
    
    def generate_block(self, block_type: str) -> None:
        # block_type = 'I'
        block = self.get_block_matrix(block_type)
        if block_type == 'O':
            self.position = [5, 1]
        else:
            self.position = [4, 1]
        self.type = block_type
        self.block_matrix = block
        self.rotation = 0
        self.touched_ground = False

    def fall_down(self, venue_matrix: list) -> bool:
        if self.is_valid_move(venue_matrix, (self.position[0], self.position[1] + 1)):
            self.position[1] += 1
            return True
        return False

    def is_valid_move(self, venue_matrix: list, pos: list) -> bool:
        row = [i for i in range(-1, len(self.block_matrix) - 1) for _ in range(len(self.block_matrix))]
        col = [i for _ in range(len(self.block_matrix)) for i in range(-1, len(self.block_matrix) - 1)]
        for i in range(len(row)):
            if self.block_matrix[1 + row[i]][1 + col[i]]:
                x = pos[0] + col[i]
                y = pos[1] + row[i]

                if x < 0 or x >= 10 or y >= 21 or y < 0 or venue_matrix[y][x]:
                    return False
        return True

    def move_left(self, venue_matrix: list) -> bool:
        new_pos = [self.position[0] - 1, self.position[1]]

        if self.is_valid_move(venue_matrix, new_pos):
            self.position = new_pos
            return True
        return False

    def move_right(self, venue_matrix: list) -> bool:
        new_pos = [self.position[0] + 1, self.position[1]]

        if self.is_valid_move(venue_matrix, new_pos):
            self.position = new_pos
            return True
        return False

    def rotate_90(self) -> None:
        self.block_matrix = [list(reversed(col)) for col in zip(*self.block_matrix)]

    def rotate_180(self) -> None:
        self.block_matrix = [row[::-1] for row in self.block_matrix[::-1]]
    
    def rotate_270(self) -> None:
        self.block_matrix = [list(row) for row in list(zip(*self.block_matrix))[::-1]]

    def rotate_90_with_wall_kick(self, venue_matrix: list, direction: int) -> bool:
        WALL_KICK_TABLE = None
        if self.type == 'I':
            WALL_KICK_TABLE = {
                (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)],
                (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)],
                (2, 3): [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)],
                (3, 0): [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)],

                (1, 0): [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)],
                (2, 1): [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)],
                (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)],
                (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)]
            }
        else:
            WALL_KICK_TABLE = {
                (0, 1): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
                (1, 2): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
                (2, 3): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
                (3, 0): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],

                (1, 0): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
                (2, 1): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
                (3, 2): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
                (0, 3): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)]
            }

        new_block = self.clone()

        new_block.rotation = (self.rotation + direction + 4) % 4
        if direction == 1:
            new_block.rotate_90()
        else:
            new_block.rotate_270()

        for offset in WALL_KICK_TABLE[(self.rotation, new_block.rotation)]:
            new_pos = [self.position[0] + offset[0], self.position[1] + offset[1]]
            if new_block.is_valid_move(venue_matrix, new_pos):
                self.rotation = new_block.rotation
                self.position = new_pos
                self.block_matrix = new_block.block_matrix
                return True
        return False
    
    def is_on_ground(self, venue_matrix: list) -> bool:
        return not self.is_valid_move(venue_matrix, (self.position[0], self.position[1] + 1))
    
    def debug(self):
        for row in self.block_matrix:
            print(row)
        print()


class Tetris:
    def __init__(self) -> None:
        self.COLOR = {
            'WHITE': (255, 255, 255),
            'LIGHT_LIGHT_GRAY': (120, 120, 120),
            'LIGHT_GRAY': (60, 60, 60),
            'DARK_GRAY': (30, 30, 30),
            'AQUA': (38, 255, 242),
            'BLUE': (38, 85, 255),
            'ORANGE': (255, 116, 38),
            'YELLOW': (251, 255, 38),
            'GREEN': (52, 255, 41),
            'PURPLE': (128, 41, 255),
            'RED': (255, 50, 50)
        }

    def start(self) -> None:
        pygame.init()
        pygame.time.set_timer(pygame.USEREVENT, 1)
        tetrisAI = TetrisAI()
        venue_matrix = [[0] * 10 for _ in range(21)]
        tetrisAI.init_setting(self.get_state_properties(venue_matrix))
        while True:
            play_mode = self.init_window()
            if play_mode == 1:
                self.game_loop(play_mode, None)
            elif play_mode == 2:
                self.game_loop(play_mode, tetrisAI)
            else:
                break
        pygame.quit()

    def init_window(self) -> int:
        BUTTON_SIZE = (240, 60)
        WINDOW_SIZE = (300, 360)

        window = Window(WINDOW_SIZE, 'Welcome to Tetris')
        running = True
        while running:
            window.event_detect()
            if window.events['quit']:
                running = False
            
            # background
            window.screen.fill(self.COLOR['DARK_GRAY'])

            # play myself button
            if window.button(self.COLOR['LIGHT_GRAY'], self.COLOR['LIGHT_LIGHT_GRAY'], (30, 30), BUTTON_SIZE):
                return 1
            window.draw_text('Play Myself', 54, self.COLOR['WHITE'], (30, 30), BUTTON_SIZE)

            # ai training button
            if window.button(self.COLOR['LIGHT_GRAY'], self.COLOR['LIGHT_LIGHT_GRAY'], (30, 150), BUTTON_SIZE):
                return 2
            window.draw_text('AI Training', 54, self.COLOR['WHITE'], (30, 150), BUTTON_SIZE)

            # quit button
            if window.button(self.COLOR['LIGHT_GRAY'], self.COLOR['LIGHT_LIGHT_GRAY'], (30, 270), BUTTON_SIZE):
                return 0
            window.draw_text('Quit', 54, self.COLOR['WHITE'], (30, 270), BUTTON_SIZE)

            pygame.display.update()

    def game_loop(self, play_mode:int, tetrisAI: TetrisAI) -> None:
        WINDOW_SIZE = (546, 686) # game (346, 686) info width 200
        BLOCK_SIZE = 32
        title = 'Tetris - Play Myself' if play_mode == 1 else f'Tetris - AI Training with DQN'
        window = Window(WINDOW_SIZE, title)
        running = True
        
        countdown = 3
        window.events['timer'] = 0
        
        while countdown:
            window.event_detect(play_mode)
            if window.events['quit']:
                return
            if window.events['timer'] >= 800:
                window.events['timer'] = 0
                countdown -= 1
            # background
            window.screen.fill(self.COLOR['DARK_GRAY'])

            # countdown
            self.draw_frame(window, BLOCK_SIZE)
            window.draw_text(str(countdown), 72, self.COLOR['WHITE'], (0, 0), WINDOW_SIZE)

            pygame.display.update()

        while True:
            # generate block
            block = Block()
            seven_bag = self.generate_seven_bag()
            block_que = [seven_bag.pop(random.randint(0, len(seven_bag) - 1)) for _ in range(5)]
            block.generate_block(self.generate_type_in_que(block_que, seven_bag))
            hold_type = None
            level = 2
            venue_matrix = [[0] * 10 for _ in range(21)]
            operate_times = 0
            window.events['timer'] = 0
            is_use_hold = False
            judge = {
                'line': 0,
                'is_T_spin': False,
                'is_T_spin_mini': False,
                'is_back_to_back': False,
                'back_to_back_flag': False,
                'is_perfect_clear': False,
                'combo': -1
            }
            score = 0
            total_reward = 0
            tetrominoes = 0
            cleared_lines = 0
            actions = []
            can_next_action = True
            if play_mode == 2:
                tetrisAI.reset_state(self.get_state_properties(venue_matrix))

            # for i in range(19, 21):
            #     for j in range(10):
            #         if not(5 <= j <= 6):
            #             venue_matrix[i][j] = 1

            while running:
                window.event_detect(play_mode)
                if window.events['quit']:
                    running = False
                    return

                # AI action
                if play_mode == 2:
                    dummy_state = {
                        'board': venue_matrix,
                        'current_block': block,
                        'block_que': block_que,
                        'hold': hold_type
                    }

                    if not len(actions):
                        code = tetrisAI.select_action(self.get_next_states(venue_matrix, block, block_que[0] if hold_type == None else hold_type))
                        actions = action_decoder(code)
                        # print(code)
                        # print(actions)
                    
                    if  can_next_action:
                        # print(actions)
                        window.events[actions[0]] = True
                        if actions[0] == 'Down_down':
                            can_next_action = False
                        actions.pop(0)

                    
                
                # control block
                is_operate = False
                if window.events['Z_down']:
                    window.events['Z_down'] = False
                    if block.rotate_90_with_wall_kick(venue_matrix, -1):
                        is_operate += 1
                        if block.type == 'T':
                            T_spin_or_mini = self.is_T_spin_or_mini(venue_matrix, block)
                            judge['is_T_spin'] = T_spin_or_mini == 1
                            judge['is_T_spin_mini'] = T_spin_or_mini == 2

                if window.events['X_down']:
                    window.events['X_down'] = False
                    if block.rotate_90_with_wall_kick(venue_matrix, 1):
                        is_operate += 1
                        if block.type == 'T':
                            T_spin_or_mini = self.is_T_spin_or_mini(venue_matrix, block)
                            judge['is_T_spin'] = T_spin_or_mini == 1
                            judge['is_T_spin_mini'] = T_spin_or_mini == 2

                if window.events['C_down']:
                    window.events['C_down'] = False
                    judge['is_T_spin'] = 0
                    judge['is_T_spin_mini'] = 0
                    if not is_use_hold:
                        is_use_hold = True
                        
                        if hold_type == None:
                            hold_type = block.type
                            block.generate_block(self.generate_type_in_que(block_que, seven_bag))
                        else:
                            temp = block.type
                            block.generate_block(hold_type)
                            hold_type = temp
                        operate_times = 0

                if window.events['Left_down']:
                    window.events['Left_down'] = False
                    judge['is_T_spin'] = False
                    judge['is_T_spin_mini'] = False
                    is_operate += block.move_left(venue_matrix)

                if window.events['Right_down']:
                    window.events['Right_down'] = False
                    judge['is_T_spin'] = False
                    judge['is_T_spin_mini'] = False
                    is_operate += block.move_right(venue_matrix)

                is_fixed_block = False
                if window.events['space_down']:
                    window.events['space_down'] = False
                    window.events['timer'] = 0
                    fall_down_count = 0
                    while (block.fall_down(venue_matrix)):
                        fall_down_count += 1
                    if fall_down_count:
                        judge['is_T_spin'] = False
                        judge['is_T_spin_mini'] = False
                    score += fall_down_count * 2
                    is_fixed_block = True

                # prosess
                else:
                    on_ground = block.is_on_ground(venue_matrix)
                    if on_ground:
                        can_next_action = True
                    if on_ground and is_operate and operate_times < 10:
                        operate_times += 1
                        window.events['timer'] = 0
                    elif on_ground and operate_times >= 10:
                        is_fixed_block = True

                    elif window.events['timer'] >= (500 if on_ground else 10 if window.events['Down_down'] else self.fall_speed(level)):
                        window.events['timer'] = 0
                        if on_ground:
                            is_fixed_block = True
                        else:
                            if block.fall_down(venue_matrix):
                                judge['is_T_spin'] = False
                                judge['is_T_spin_mini'] = False

                                if window.events['Down_down']:
                                    score += 1

                if is_fixed_block:
                    block, operate_times, is_use_hold = self.fixed_block_and_init(venue_matrix, block, block_que, seven_bag, judge)
                    score += self.get_score(judge)
                    tetrominoes += 1
                    cleared_lines += judge['line']


                # screen update
                window.screen.fill(self.COLOR['DARK_GRAY'])
                self.draw_frame(window, BLOCK_SIZE)
                self.draw_venue_matrix(window, venue_matrix)
                if not self.draw_block(window, block, venue_matrix):
                    running = False
                
                self.draw_info(window, block_que, hold_type, score, running, tetrisAI.episode if play_mode == 2 else None)
                pygame.display.update()

                # AI reward
                if play_mode == 2 and is_fixed_block:
                    reward = 1 + (judge['line'] ** 2) * 10
                    total_reward += reward
                    if not running:
                        reward -= 2
                    tetrisAI.update(reward, not running, total_reward, tetrominoes, cleared_lines, score)

            window.events['timer'] = 0
            while True:
                window.event_detect()
                if window.events['quit']:
                    running = False
                    return
                
                if play_mode == 2 and window.events['timer'] > 100:
                    running = True
                    break

    def get_next_states(self, venue_matrix: list, current_block: Block, hold_type: str) -> dict:
        state = {}

        hold_block = Block()
        hold_block.generate_block(hold_type)
        blocks = [current_block, hold_block]
        movements = [None, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1]

        for i, block in enumerate(blocks):
            temp_block = block.clone()
            rotations = None
            if block.type == 'O':
                rotations = [None]
            elif block.type == 'I' or block.type == 'S' or block.type == 'Z':
                rotations = [None, 1]
            else:
                rotations = [None, 1, 1, 0, -1]

            action = [i, 0, 0] # is_use_hold, rotate, move
            for rotation in rotations:
                temp_block.position = block.position[:]
                action[2] = 0
                if rotation == 0:
                    temp_block.rotation = 0
                    temp_block.block_matrix = [row[:] for row in block.block_matrix]
                    action[1] = 0
                    continue
                if rotation != None:
                    if not temp_block.rotate_90_with_wall_kick(venue_matrix, rotation):
                        continue
                    action[1] += rotation

                for movement in movements:
                    if movement == 0:
                        temp_block.position = block.position[:]
                        action[2] = 0
                        continue
                    if movement != None:
                        if movement == 1 and not temp_block.move_right(venue_matrix):
                            continue
                        if movement == -1 and not temp_block.move_left(venue_matrix):
                            continue
                        action[2] += movement

                    while temp_block.fall_down(venue_matrix):
                        pass
                    temp_venue_matrix = [row[:] for row in venue_matrix]
                    self.fixed_block(temp_venue_matrix, temp_block)
                    # print(action)
                    # self.debug(temp_venue_matrix)
                    state[tuple(action)] = self.get_state_properties(temp_venue_matrix)

        return state
                
    def get_state_properties(self, venue_matrix: list) -> torch.FloatTensor:
        line_clears = self.eliminate_blocks(venue_matrix)
        holes = self.get_hole_count(venue_matrix)
        bumpiness, height = self.get_bumpiness_and_height(venue_matrix)
        # print(holes, bumpiness, height)
        return torch.FloatTensor([line_clears, holes, bumpiness, height])

    def get_hole_count(self, venue_matrix: list) -> int:
        count = 0
        for col in zip(*venue_matrix):
            row = 0
            while row < 21 and col[row] == 0:
                row += 1
            count += len([x for x in col[row + 1:] if x == 0])
        return count
    
    def get_bumpiness_and_height(self, venue_matrix: list) -> tuple:
        venue_matrix = numpy.array(venue_matrix)
        mask = venue_matrix != 0
        invert_heights = numpy.where(mask.any(axis=0), numpy.argmax(mask, axis=0), 21)
        heights = 21 - invert_heights
        total_height = numpy.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = numpy.abs(currs - nexts)
        total_bumpiness = numpy.sum(diffs)
        return total_bumpiness, total_height

    def draw_frame(self, window: Window, block_size: int) -> None:
        for i in range(21):
            interval = block_size + 2
            window.draw(self.COLOR['LIGHT_GRAY'], (2, i * interval + 2), (interval * 10 - 2, 2))
        for i in range(11):
            window.draw(self.COLOR['LIGHT_GRAY'], (i * interval + 2, 2), (2, interval * 20 - 2))

    def fall_speed(self, level: int) -> int:
        DROP_SPEEDS = [1000, 793, 617, 473, 355, 262, 190, 135, 93, 66, 50]
        interval = DROP_SPEEDS[min(level, len(DROP_SPEEDS) - 1)]
        return interval

    def draw_venue_matrix(self, window: Window, venue_matrix: list) -> None:
        BLOCK_COLOR = ['AQUA', 'BLUE', 'ORANGE', 'YELLOW', 'GREEN', 'PURPLE', 'RED']
        for i in range(1, 21):
            for j in range(10):
                if venue_matrix[i][j]:
                    window.draw(self.COLOR[BLOCK_COLOR[venue_matrix[i][j] - 1]], (4 + 34 * j, 4 + 34 * (i - 1)), (32, 32))
                else:
                    window.draw(self.COLOR['DARK_GRAY'], (4 + 34 * j, 4 + 34 * (i - 1)), (32, 32))

    def draw_block(self, window: Window, block: Block, venue_matrix: list) -> bool:
        BLOCK_COLOR = {
            'I': 'AQUA',
            'J': 'BLUE',
            'L': 'ORANGE',
            'O': 'YELLOW',
            'S': 'GREEN',
            'T': 'PURPLE',
            'Z': 'RED'
        }
        row = [i for i in range(-1, len(block.block_matrix) - 1) for _ in range(len(block.block_matrix))]
        col = [i for _ in range(len(block.block_matrix)) for i in range(-1, len(block.block_matrix) - 1)]

        for i in range(len(row)):
            if block.block_matrix[1 + row[i]][1 + col[i]] and\
                venue_matrix[block.position[1] + row[i]][block.position[0] + col[i]]:
                return False
            
        for i in range(len(row)):
            if block.block_matrix[1 + row[i]][1 + col[i]]:
                window.draw(self.COLOR[BLOCK_COLOR[block.type]], (4 + 34 * (block.position[0] + col[i]), 4 + 34 * (block.position[1] + row[i] - 1)), (32, 32))
        return True

    def fixed_block(self, venue_matrix: list, block: Block) -> None:
        row = [i for i in range(-1, len(block.block_matrix) - 1) for _ in range(len(block.block_matrix))]
        col = [i for _ in range(len(block.block_matrix)) for i in range(-1, len(block.block_matrix) - 1)]
        for i in range(len(row)):
            if block.block_matrix[1 + row[i]][1 + col[i]] and \
                0 <= block.position[1] + row[i] < len(venue_matrix) and \
                0 <= block.position[0] + col[i] < len(venue_matrix[0]):
                venue_matrix[block.position[1] + row[i]][block.position[0] + col[i]] = block.block_matrix[1 + row[i]][1 + col[i]]

    def eliminate_blocks(self, venue_matrix: list) -> int:
        full_blocks_row = []
        for i, row in enumerate(venue_matrix):
            if row.count(0):
                continue
            full_blocks_row.append(i)
        
        for i, row in enumerate(full_blocks_row):
            venue_matrix.pop(row - i)

        for _ in range(len(full_blocks_row)):
            venue_matrix.insert(0, [0] * 10)

        return len(full_blocks_row)
    
    def draw_info(self, window: Window, block_que: list, hold_type: str, score: int, running: bool, episode) -> None:
        BLOCK_COLOR = {
            'I': 'AQUA',
            'J': 'BLUE',
            'L': 'ORANGE',
            'O': 'YELLOW',
            'S': 'GREEN',
            'T': 'PURPLE',
            'Z': 'RED'
        }

        window.draw_text('Hold', 36, self.COLOR['WHITE'], (342, 0), (204, 50))
        if hold_type != None:
            window.draw_text(hold_type, 72, self.COLOR[BLOCK_COLOR[hold_type]], (342, 50), (204, 50))

        window.draw_text('Block Queue', 36, self.COLOR['WHITE'], (342, 100), (204, 50))
        for i, block_type in enumerate(block_que):
            window.draw_text(block_type, 72, self.COLOR[BLOCK_COLOR[block_type]], (342, 100 + 70 * i + 70), (204, 50))

        window.draw_text('Score', 36, self.COLOR['WHITE'], (342, 520), (204, 50))
        window.draw_text(str(score), 36, self.COLOR['WHITE'], (342, 570), (204, 50))

        if not running:
            window.draw_text('Game Over', 36, self.COLOR['RED'], (342, 620), (204, 50))
        elif episode != None:
            window.draw_text(f'Episode: {episode}', 36, self.COLOR['WHITE'], (342, 620), (204, 50))

    def generate_type_in_que(self, block_que: list, seven_bag: list) -> str:
        if not len(seven_bag):
            for block_type in self.generate_seven_bag():
                seven_bag.append(block_type)
        block_que.append(seven_bag.pop(random.randint(0, len(seven_bag) - 1)))
        return block_que.pop(0)
    
    def generate_seven_bag(self) -> list:
        return ['I', 'J', 'L', 'O', 'S', 'T', 'Z']
        # return ['L', 'I', 'S', 'Z'] * 2
        # return ['Z', 'S'] * 3

    def fixed_block_and_init(self, venue_matrix: list, block: Block, block_que: list, seven_bag: list, judge: dict):
        self.fixed_block(venue_matrix, block)
        block.generate_block(self.generate_type_in_que(block_que, seven_bag))
        operate_times = 0
        is_use_hold = False
        judge['line'] = self.eliminate_blocks(venue_matrix)
        judge['is_perfect_clear'] = self.is_perfect_clear(venue_matrix)
        return block, operate_times, is_use_hold
    
    def is_perfect_clear(self, venue_matrix: list) -> bool:
        return all(all(col == 0 for col in row) for row in venue_matrix)
    
    def get_score(self, judge: dict) -> int:
        if judge['back_to_back_flag']:
            judge['is_back_to_back'] = True

        if judge['line']:
            judge['combo'] += 1
        else:
            judge['combo'] = -1

        if judge['line'] == 4 or (judge['is_T_spin'] or judge['is_T_spin_mini']) and judge['line']:
            judge['back_to_back_flag'] = True
        elif judge['line']:
            judge['back_to_back_flag'] = False
            judge['is_back_to_back'] = False

        score = 50 * judge['combo'] if judge['combo'] >= 1 else 0

        if judge['is_perfect_clear']:
            if judge['line'] == 1:
                score += 800
            elif judge['line'] == 2:
                score += 1200
            elif judge['line'] == 3:
                score += 1800
            elif judge['line'] == 4 and not judge['is_back_to_back']:
                score += 2000
            else:
                score += 3200
        else:
            if judge['is_T_spin_mini']:
                score += 100 * 2**judge['line']
            elif judge['is_T_spin']:
                score += 400 * (judge['line'] + 1)
            elif judge['line'] == 1:
                score += 100
            elif judge['line'] == 2:
                score += 300
            elif judge['line'] == 3:
                score += 500
            elif judge['line'] == 4:
                score += 800

            if judge['is_back_to_back']:
                score *= 1.5
                score = int(score)

        # debug
        # print_flag = False
        # if judge['is_T_spin']:
        #     print('T-Spin ', end='')
        #     print_flag = True
        # elif judge['is_T_spin_mini']:
        #     print('T-Spin Mini ', end='')
        #     print_flag = True

        # if judge['line'] == 1:
        #     print('Single', end='')
        #     print_flag = True
        # elif judge['line'] == 2:
        #     print('Double', end='')
        #     print_flag = True
        # elif judge['line'] == 3:
        #     print('Trible', end='')
        #     print_flag = True
        # elif judge['line'] == 4:
        #     print('Tetris', end='')
        #     print_flag = True

        # if judge['line'] and judge['is_back_to_back']:
        #     print(' Back To Back', end='')
        #     print_flag = True

        # if judge['combo'] >= 1:
        #     print(' Combo:', judge['combo'], end='')
        #     print_flag = True

        # if judge['is_perfect_clear']:
        #     print(' Perfect Clear')
        #     print_flag = True

        # if print_flag:
        #     print(f'\nScore: {score}\n')

        judge['is_T_spin'] = False
        judge['is_T_spin_mini'] = False
        return score

    def is_T_spin_or_mini(self, venue_matrix: list, block: Block) -> int: # 1 T-Spin, 2 Mini
        hole_count = 0
        for row in [-1, 1]:
            for col in [-1, 1]:
                if block.position[1] + row >= len(venue_matrix) or\
                    block.position[1] + row < 0 or\
                    block.position[0] + col >= len(venue_matrix[0]) or\
                    block.position[0] + col < 0 or\
                    venue_matrix[block.position[1] + row][block.position[0] + col]:
                        hole_count += 1
        
        if hole_count < 3:
            return 0
        
        CHECK_POS = [(-1, -1), (1, -1), (1, 1), (-1, 1)] # (x, y)
        offset1 = (CHECK_POS[block.rotation][0], CHECK_POS[block.rotation][1])
        offset2 = (CHECK_POS[(block.rotation + 1) % 4][0], CHECK_POS[(block.rotation + 1) % 4][1])
        if venue_matrix[block.position[1] + offset1[1]][block.position[0] + offset1[0]] \
            and venue_matrix[block.position[1] + offset2[1]][block.position[0] + offset2[0]]:
            return 1
        return 2

    def debug(self, matrix: list) -> None:
        for row in matrix:
            print(row)
        print()


if __name__ == '__main__':
    Tetris().start()