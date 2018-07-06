import pygame

from  learn.chapter12.restore_points.restore_point_1_ship_moves.settings import Settings
from learn.chapter12.restore_points.restore_point_1_ship_moves.ship import Ship
import learn.chapter12.restore_points.restore_point_1_ship_moves.game_functions as gf

def run_game():
    # Initialize pygame, settings, and screen object.
    pygame.init()
    ai_settings = Settings()
    screen = pygame.display.set_mode(
        (ai_settings.screen_width, ai_settings.screen_height))
    pygame.display.set_caption("Alien Invasion")

    # Set the background color.
    bg_color = (230, 230, 230)

    # Make a ship.
    ship = Ship(ai_settings, screen)

    # Start the main loop for the game.
    while True:
        gf.check_events(ship)
        ship.update()
        gf.update_screen(ai_settings, screen, ship)

run_game()
