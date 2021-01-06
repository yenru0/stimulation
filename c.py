import pygame
import pymunk
from pymunk import Vec2d as Vector2

G_ = 500


class AbstractPoint:

    def __init__(self, _space: pymunk.Space, init_pos: Vector2, mass=1, k=10, color=(255, 0, 0)):
        self.space = _space
        self.body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 1, 0))
        self.body.position = init_pos
        self.space.add(self.body)
        self.k = k
        self.color = color

    def draw(self, _surface: pygame.Surface):
        pygame.draw.circle(_surface, self.color, pymunk.pygame_util.to_pygame(self.body.position, _surface), 3)

    def applyForce(self, force: Vector2):
        self.body.apply_force_at_local_point(force, (0, 0))

    def attract(self, _ap: 'AbstractPoint'):
        force = _ap.body.position - self.body.position
        distance_sqrd = force.get_length_sqrd()

        force = force.normalized()
        strength = -(G_ * _ap.body.mass * self.body.mass) / distance_sqrd
        _ap.applyForce(strength * force)


if __name__ == "__main__":
    import pymunk.pygame_util

    pygame.init()

    time_step = 1 / 60
    TARGET_FPS = 120
    clock = pygame.time.Clock()

    surface: pygame.Surface = pygame.display.set_mode((500, 500))

    space = pymunk.Space()
    draw_options = pymunk.pygame_util.DrawOptions(surface)

    isExit = False
    fc = 0
    # points = [AbstractPoint(space, Vector2(random.uniform(0, 500), random.uniform(0, 500))) for _ in range(100)]
    points = [AbstractPoint(space, Vector2(250, 250), -1, color=(0, 0, 255)), AbstractPoint(space, Vector2(200, 250))]

    while not isExit:
        surface.fill((255, 255, 255))

        space.debug_draw(draw_options)
        for ap in points:
            ap.draw(surface)
        for ap1 in points:
            for ap2 in points:
                if ap1 != ap2:
                    ap1.attract(ap2)

        space.step(time_step)
        for event in pygame.event.get(pygame.QUIT):
            print("end")
            isExit = True
        if fc == 3000:
            isExit = True

        fc += 1
        clock.tick(TARGET_FPS)
        pygame.display.flip()
    pygame.quit()
