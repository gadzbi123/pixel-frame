"""Animated ASCII woodland, using only Python's standard library."""
import curses
import math
import random
import time


def phase_at(seconds, cycle):
    phase = (seconds / cycle) % 1
    return phase, ('DAWN', 'DAY', 'DUSK', 'NIGHT')[int(phase * 4)]


class TextCanvas:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.cells = [[(' ', 1) for _ in range(width)] for _ in range(height)]

    def text(self, x, y, value, color=1):
        for i, char in enumerate(value):
            if 0 <= x+i < self.width and 0 <= y < self.height:
                self.cells[y][x+i] = (char, color)

    def sprite(self, x, y, lines, color):
        for row, line in enumerate(lines):
            for col, char in enumerate(line):
                if char != ' ':
                    self.text(x+col, y+row, char, color)


def render_text(width, height, seconds, seed=1, cycle=120):
    c = TextCanvas(width, height)
    if width < 40 or height < 18:
        c.text(0, 0, 'Woodland needs at least 40 columns x 18 rows.', 4)
        c.text(0, 2, 'Use a smaller console font. Q quits.', 1)
        return c
    rng = random.Random(seed)
    phase, name = phase_at(seconds, cycle)
    night = name == 'NIGHT'
    ground = height - 7
    sky_height = max(4, ground - 7)
    if night:
        for _ in range(width//2):
            x, y = rng.randrange(width), rng.randrange(2, sky_height+2)
            c.text(x, y, '+' if int(seconds*2+x)%13 == 0 else '.', 7)
    orb_x = int(width * (.12 + .7*((phase*2) % 1)))
    orb_y = 3 + int(2*abs(math.cos(phase*2*math.pi)))
    c.sprite(orb_x, orb_y, [' _ ', '( )', ' - '] if night else ['\\ | /', '- O -', '/ | \\'], 7 if night else 4)
    for n in range(3):
        x = int((n*width/3 + seconds*.7) % (width+20))-12
        y = 3+n%2*3
        c.sprite(x, y, ['   .--.   ', '.-(    ). ', " '------' "], 7 if night else 1)
    # Distant ridgeline, deliberately quieter than the animals.
    for x in range(width):
        y = ground-5-int(2+2*math.sin(x*.12)+math.sin(x*.27))
        c.text(x, y, '/' if math.cos(x*.12)>0 else '\\', 5)
    for base, size in [(int(width*.10), 7), (int(width*.25), 5), (int(width*.78), 6), (int(width*.91), 8)]:
        for row in range(size):
            sway = round(math.sin(seconds*1.5+base+row*.18)*(1-row/size))
            half = 1+row//2
            c.text(base-half+sway, ground-size+row, '/'+'^'*(half*2-1)+'\\', 2)
        c.text(base-1, ground, '| |', 6)
    # A warm little home, smoke rises even while the animals rest.
    home = width//2-6
    c.sprite(home, ground-5, ['    /\\    ', '   /==\\   ', '  /====\\  ', '  | [] |  ', '  | __ |  ', '__|_||_|__'], 6)
    c.text(home+4, ground-2, '[]', 4)
    for i in range(3):
        c.text(home+7+round(math.sin(seconds+i)), ground-7-i, '~', 7)
    for x in range(width):
        c.text(x, ground+1, '_' if x%3 else ',', 2)
        if x%5 == 0:
            c.text(x, ground, "'" if math.sin(seconds*2+x)>0 else '`', 2)
    # Birds flap, cross the view, then return in another formation.
    for n in range(3):
        x = int((seconds*4+n*7) % (width+28))-14
        y = 3+n+round(math.sin(seconds*.7+n))
        c.text(x, y, '\\_/' if int(seconds*4+n)%2 else '/~\\', 7 if night else 4)
    # Fox walks in, sniffs, sits, and leaves. Keep a full-size subject visible.
    act = seconds % 36
    target = max(2, width//4-5)
    fox_x = round(-12+(target+12)*min(act/8, 1)) if act<24 else round(target+(width+12-target)*(act-24)/12)
    if 12 <= act < 21:
        fox = [' /\\_/\\', '( o.o )', ' /| |\\___', '(_|_|____)']
    else:
        fox = [' /\\_/\\', '( o.o )____   /\\', ' /       __\\_/ /', '  /_/ /_/  \\__/']
        if int(seconds*5)%2:
            fox[-1] = '  \\_\\ \\_\\  \\__/'
    for row, line in enumerate(fox):
        c.text(fox_x, ground-2+row, ' '*len(line))
    c.sprite(fox_x, ground-2, fox, 6)
    # Rabbit hops near the stream, with pauses and blinking eyes.
    rabbit_x = int(width*.67)+round(3*math.sin(seconds*.28))
    hop = round(max(0, math.sin(seconds*3))*2) if int(seconds)%12<5 else 0
    c.sprite(rabbit_x, ground-2-hop, ['(\\ /)', '(o.o)' if int(seconds)%7 else '(-.-)', '(")(")'], 1)
    for y in range(ground+3, height-2):
        for x in range(width):
            c.text(x, y, '~' if (x+int(seconds*3)+y*3)%11<4 else ' ', 3)
    if night:
        for n in range(8):
            if math.sin(seconds*2+n)>0:
                c.text(int((n*17+seconds)%width), ground-2-round(2*math.sin(n+seconds)), '*', 4)
    c.text(2, 0, f' {name}  /  WOODLAND ', 4)
    c.text(2, height-1, 'Q quit   SPACE pause   N next time of day', 7)
    return c


def run_tty(seed=1, fps=12, cycle=120):
    def loop(screen):
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        screen.nodelay(True)
        has_color = curses.has_colors()
        if has_color:
            curses.start_color()
            for i, fg in enumerate([curses.COLOR_WHITE, curses.COLOR_GREEN, curses.COLOR_CYAN, curses.COLOR_YELLOW, curses.COLOR_BLUE, curses.COLOR_YELLOW, curses.COLOR_WHITE], 1):
                curses.init_pair(i, fg, curses.COLOR_BLACK)
        last_period = None
        elapsed, previous, paused = 0., time.monotonic(), False
        while True:
            now = time.monotonic()
            if not paused:
                elapsed += now-previous
            previous = now
            key = screen.getch()
            if key in (ord('q'), ord('Q'), 27):
                break
            if key == ord(' '):
                paused = not paused
            if key in (ord('n'), ord('N')):
                elapsed += cycle/4
            period = phase_at(elapsed, cycle)[1]
            if has_color and period != last_period:
                # Standard Linux-console colors; no truecolor or Unicode needed.
                bg = curses.COLOR_BLUE if period in ('DAY', 'DAWN') else curses.COLOR_BLACK
                for i, fg in enumerate([curses.COLOR_WHITE, curses.COLOR_GREEN, curses.COLOR_CYAN, curses.COLOR_YELLOW, curses.COLOR_CYAN, curses.COLOR_YELLOW, curses.COLOR_WHITE], 1):
                    curses.init_pair(i, fg, bg)
                last_period = period
            height, width = screen.getmaxyx()
            canvas = render_text(width, height, elapsed, seed, cycle)
            for y, row in enumerate(canvas.cells):
                for x, (char, color) in enumerate(row):
                    attr = curses.color_pair(color) if has_color else 0
                    if color in (1, 2, 4, 6):
                        attr |= curses.A_BOLD
                    try:
                        screen.addch(y, x, char, attr)
                    except curses.error:
                        pass  # Includes curses' bottom-right-cell convention.
            screen.refresh()
            time.sleep(max(0, 1/fps-(time.monotonic()-now)))
    curses.wrapper(loop)
