from __future__ import division
import random
import sys
import pygame
import scipy.misc
import numpy as np
from six.moves import xrange
from itertools import cycle
from pygame.locals import *
from PIL import Image

FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512
# amount by which base can maximum shift to left
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

PLAYER_IMAGES = (
	"assets/sprites/tippy-upflap.png",
	"assets/sprites/tippy-midflap.png",
)

BACKGROUND_IMAGE = "assets/sprites/background.png"
PIPE_IMAGE = "assets/sprites/pipe.png"

# list of pipes
PIPES_LIST = (
	"assets/sprites/pipe.png",
)


def main():
	global SCREEN, FPSCLOCK
	pygame.init()
	FPSCLOCK = pygame.time.Clock()
	SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
	pygame.display.set_caption("Flappy Tippy")

	# numbers sprites for score display
	IMAGES["numbers"] = (
		pygame.image.load("assets/sprites/0.png").convert_alpha(),
		pygame.image.load("assets/sprites/1.png").convert_alpha(),
		pygame.image.load("assets/sprites/2.png").convert_alpha(),
		pygame.image.load("assets/sprites/3.png").convert_alpha(),
		pygame.image.load("assets/sprites/4.png").convert_alpha(),
		pygame.image.load("assets/sprites/5.png").convert_alpha(),
		pygame.image.load("assets/sprites/6.png").convert_alpha(),
		pygame.image.load("assets/sprites/7.png").convert_alpha(),
		pygame.image.load("assets/sprites/8.png").convert_alpha(),
		pygame.image.load("assets/sprites/9.png").convert_alpha()
	)

	# game over sprite
	IMAGES["gameover"] = pygame.image.load("assets/sprites/gameover.png").convert_alpha()
	# message sprite for welcome screen
	IMAGES["message"] = pygame.image.load("assets/sprites/message.png").convert_alpha()
	# base (ground) sprite
	IMAGES["base"] = pygame.image.load("assets/sprites/base.png").convert_alpha()

	# sounds
	if "win" in sys.platform:
		soundExt = ".wav"
	else:
		soundExt = ".ogg"

	SOUNDS["die"]    = pygame.mixer.Sound("assets/audio/die" + soundExt)
	SOUNDS["hit"]    = pygame.mixer.Sound("assets/audio/hit" + soundExt)
	SOUNDS["point"]  = pygame.mixer.Sound("assets/audio/point" + soundExt)
	SOUNDS["swoosh"] = pygame.mixer.Sound("assets/audio/swoosh" + soundExt)
	SOUNDS["wing"]   = pygame.mixer.Sound("assets/audio/wing" + soundExt)

	IMAGES["background"] = pygame.image.load(BACKGROUND_IMAGE).convert()

	IMAGES["player"] = (
		pygame.image.load(PLAYER_IMAGES[0]).convert_alpha(),
		pygame.image.load(PLAYER_IMAGES[1]).convert_alpha(),
	)

	IMAGES["pipe"] = (
		pygame.transform.rotate(pygame.image.load(PIPE_IMAGE).convert_alpha(), 180),
		pygame.image.load(PIPE_IMAGE).convert_alpha(),
	)

	# hismask for pipes
	HITMASKS["pipe"] = (
		get_hitmask(IMAGES["pipe"][0]),
		get_hitmask(IMAGES["pipe"][1]),
	)

	# hitmask for player
	HITMASKS["player"] = (
		get_hitmask(IMAGES["player"][0]),
		get_hitmask(IMAGES["player"][1]),
		# get_hitmask(IMAGES["player"][2]),
	)
	while True:
		movementInfo = show_welcome_animation()
		crashInfo = run_main_game(movementInfo)
		show_game_over_screen(crashInfo)


def show_welcome_animation():
	"""Shows welcome screen animation of flappy bird"""
	# index of player to blit on screen
	playerIndex = 0
	playerIndexGen = cycle([0, 1])
	# iterator used to change playerIndex after every 5th iteration
	loopIter = 0

	playerx = int(SCREENWIDTH * 0.2)
	playery = int((SCREENHEIGHT - IMAGES["player"][0].get_height()) / 2)

	messagex = int((SCREENWIDTH - IMAGES["message"].get_width()) / 2)
	messagey = int(SCREENHEIGHT * 0.12)

	basex = 0
	# amount by which base can maximum shift to left
	baseShift = IMAGES["base"].get_width() - IMAGES["background"].get_width()

	# player shm for up-down motion on welcome screen
	playerShmVals = {"val": 0, "dir": 1}

	while True:
		for event in pygame.event.get():
			if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
				pygame.quit()
				sys.exit()
			if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
				# make first flap sound and return values for run_main_game
				SOUNDS["wing"].play()
				return {
					"playery": playery + playerShmVals["val"],
					"basex": basex,
					"playerIndexGen": playerIndexGen,
				}

		# adjust playery, playerIndex, basex
		if (loopIter + 1) % 5 == 0:
			playerIndex = next(playerIndexGen)
		loopIter = (loopIter + 1) % 30
		basex = -((-basex + 4) % baseShift)
		player_shm(playerShmVals)

		# draw sprites
		SCREEN.blit(IMAGES["background"], (0,0))
		SCREEN.blit(IMAGES["player"][playerIndex], (playerx, playery + playerShmVals["val"]))
		SCREEN.blit(IMAGES["message"], (messagex, messagey))
		SCREEN.blit(IMAGES["base"], (basex, BASEY))

		pygame.display.update()
		FPSCLOCK.tick(FPS)

def run_main_game(movementInfo):
	score = playerIndex = loopIter = 0
	playerIndexGen = movementInfo["playerIndexGen"]
	playerx, playery = int(SCREENWIDTH * 0.2), movementInfo["playery"]

	basex = movementInfo["basex"]
	baseShift = IMAGES["base"].get_width() - IMAGES["background"].get_width()

	# get 2 new pipes to add to upperPipes lowerPipes list
	newPipe1 = get_random_pipe()
	newPipe2 = get_random_pipe()

	# list of upper pipes
	upperPipes = [
		{"x": SCREENWIDTH + 200, "y": newPipe1[0]["y"]},
		{"x": SCREENWIDTH + 200 + (SCREENWIDTH / 2), "y": newPipe2[0]["y"]},
	]

	# list of lowerpipe
	lowerPipes = [
		{"x": SCREENWIDTH + 200, "y": newPipe1[1]["y"]},
		{"x": SCREENWIDTH + 200 + (SCREENWIDTH / 2), "y": newPipe2[1]["y"]},
	]

	pipeVelX = -4

	# player velocity, max velocity, downward accleration, accleration on flap
	playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
	playerMaxVelY =  10   # max vel along Y, max descend speed
	playerMinVelY =  -8   # min vel along Y, max ascend speed
	playerAccY    =   1   # players downward accleration
	playerFlapAcc =  -9   # players speed on flapping
	playerFlapped = False # True when player flaps

	while True:
		for event in pygame.event.get():
			if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
				pygame.quit()
				sys.exit()
			if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
				if playery > -2 * IMAGES["player"][0].get_height():
					playerVelY = playerFlapAcc
					playerFlapped = True
					SOUNDS["wing"].play()

		# check for crash here
		crashTest = check_crash({"x": playerx, "y": playery, "index": playerIndex},
							   upperPipes, lowerPipes)
		if crashTest[0]:
			return {
				"y": playery,
				"groundCrash": crashTest[1],
				"basex": basex,
				"upperPipes": upperPipes,
				"lowerPipes": lowerPipes,
				"score": score,
				"playerVelY": playerVelY,
			}

		# check for score
		playerMidPos = playerx + IMAGES["player"][0].get_width() / 2
		for pipe in upperPipes:
			pipeMidPos = pipe["x"] + IMAGES["pipe"][0].get_width() / 2
			if pipeMidPos <= playerMidPos < pipeMidPos + 4:
				score += 1
				SOUNDS["point"].play()

		# playerIndex basex change
		if (loopIter + 1) % 3 == 0:
			playerIndex = next(playerIndexGen)
		loopIter = (loopIter + 1) % 30
		basex = -((-basex + 100) % baseShift)

		# player's movement
		if playerVelY < playerMaxVelY and not playerFlapped:
			playerVelY += playerAccY
		if playerFlapped:
			playerFlapped = False
		playerHeight = IMAGES["player"][playerIndex].get_height()
		playery += min(playerVelY, BASEY - playery - playerHeight)

		# move pipes to left
		for uPipe, lPipe in zip(upperPipes, lowerPipes):
			uPipe["x"] += pipeVelX
			lPipe["x"] += pipeVelX

		# add new pipe when first pipe is about to touch left of screen
		if 0 < upperPipes[0]["x"] < 5:
			newPipe = get_random_pipe()
			upperPipes.append(newPipe[0])
			lowerPipes.append(newPipe[1])

		# remove first pipe if its out of the screen
		if upperPipes[0]["x"] < -IMAGES["pipe"][0].get_width():
			upperPipes.pop(0)
			lowerPipes.pop(0)

		# draw sprites
		SCREEN.blit(IMAGES["background"], (0,0))

		for uPipe, lPipe in zip(upperPipes, lowerPipes):
			SCREEN.blit(IMAGES["pipe"][0], (uPipe["x"], uPipe["y"]))
			SCREEN.blit(IMAGES["pipe"][1], (lPipe["x"], lPipe["y"]))

		SCREEN.blit(IMAGES["base"], (basex, BASEY))
		# print score so player overlaps the score
		show_score(score)
		SCREEN.blit(IMAGES["player"][playerIndex], (playerx, playery))

		pygame.display.update()
		FPSCLOCK.tick(FPS)


		buf = SCREEN.get_buffer()
		image = Image.frombytes("RGBA",SCREEN.get_size(),buf.raw)
		del buf
		pixel = np.asarray(image, dtype=np.uint8)
		pixel = 0.2126 * pixel[..., 2] + 0.7152 * pixel[..., 1] + 0.0722 * pixel[..., 0]
		pixel = scipy.misc.imresize(pixel, size=(96, 72), interp="bilinear")
		pixel = pixel[0:72, 0:72]
		# Image.fromarray(pixel).convert("RGB").save("screen.bmp")

def show_game_over_screen(crashInfo):
	"""crashes the player down ans shows gameover image"""
	score = crashInfo["score"]
	playerx = SCREENWIDTH * 0.2
	playery = crashInfo["y"]
	playerHeight = IMAGES["player"][0].get_height()
	playerVelY = crashInfo["playerVelY"]
	playerAccY = 2

	basex = crashInfo["basex"]

	upperPipes, lowerPipes = crashInfo["upperPipes"], crashInfo["lowerPipes"]

	# play hit and die sounds
	SOUNDS["hit"].play()
	if not crashInfo["groundCrash"]:
		SOUNDS["die"].play()

	while True:
		for event in pygame.event.get():
			if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
				pygame.quit()
				sys.exit()
			if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
				if playery + playerHeight >= BASEY - 1:
					return

		# player y shift
		if playery + playerHeight < BASEY - 1:
			playery += min(playerVelY, BASEY - playery - playerHeight)

		# player velocity change
		if playerVelY < 15:
			playerVelY += playerAccY

		# draw sprites
		SCREEN.blit(IMAGES["background"], (0,0))

		for uPipe, lPipe in zip(upperPipes, lowerPipes):
			SCREEN.blit(IMAGES["pipe"][0], (uPipe["x"], uPipe["y"]))
			SCREEN.blit(IMAGES["pipe"][1], (lPipe["x"], lPipe["y"]))

		SCREEN.blit(IMAGES["base"], (basex, BASEY))
		show_score(score)
		SCREEN.blit(IMAGES["player"][1], (playerx,playery))

		FPSCLOCK.tick(FPS)
		pygame.display.update()

def player_shm(playerShm):
	"""oscillates the value of playerShm["val"] between 8 and -8"""
	if abs(playerShm["val"]) == 8:
		playerShm["dir"] *= -1

	if playerShm["dir"] == 1:
		 playerShm["val"] += 1
	else:
		playerShm["val"] -= 1

def get_random_pipe():
	"""returns a randomly generated pipe"""
	# y of gap between upper and lower pipe
	gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
	gapY += int(BASEY * 0.2)
	pipeHeight = IMAGES["pipe"][0].get_height()
	pipeX = SCREENWIDTH + 10

	return [
		{"x": pipeX, "y": gapY - pipeHeight},  # upper pipe
		{"x": pipeX, "y": gapY + PIPEGAPSIZE}, # lower pipe
	]

def show_score(score):
	"""displays score in center of screen"""
	scoreDigits = [int(x) for x in list(str(score))]
	totalWidth = 0 # total width of all numbers to be printed

	for digit in scoreDigits:
		totalWidth += IMAGES["numbers"][digit].get_width()

	Xoffset = (SCREENWIDTH - totalWidth) / 2

	for digit in scoreDigits:
		SCREEN.blit(IMAGES["numbers"][digit], (Xoffset, SCREENHEIGHT * 0.1))
		Xoffset += IMAGES["numbers"][digit].get_width()

def check_crash(player, upperPipes, lowerPipes):
	"""returns True if player collders with base or pipes."""
	pi = player["index"]
	player["w"] = IMAGES["player"][0].get_width()
	player["h"] = IMAGES["player"][0].get_height()

	# if player crashes into ground
	if player["y"] + player["h"] >= BASEY - 1:
		return [True, True]
	else:

		playerRect = pygame.Rect(player["x"], player["y"],
					  player["w"], player["h"])
		pipeW = IMAGES["pipe"][0].get_width()
		pipeH = IMAGES["pipe"][0].get_height()

		for uPipe, lPipe in zip(upperPipes, lowerPipes):
			# upper and lower pipe rects
			uPipeRect = pygame.Rect(uPipe["x"], uPipe["y"], pipeW, pipeH)
			lPipeRect = pygame.Rect(lPipe["x"], lPipe["y"], pipeW, pipeH)

			# player and upper/lower pipe hitmasks
			pHitMask = HITMASKS["player"][pi]
			uHitmask = HITMASKS["pipe"][0]
			lHitmask = HITMASKS["pipe"][1]

			# if bird collided with upipe or lpipe
			uCollide = pixel_collision(playerRect, uPipeRect, pHitMask, uHitmask)
			lCollide = pixel_collision(playerRect, lPipeRect, pHitMask, lHitmask)

			if uCollide or lCollide:
				return [True, False]

	return [False, False]

def pixel_collision(rect1, rect2, hitmask1, hitmask2):
	rect = rect1.clip(rect2)

	if rect.width == 0 or rect.height == 0:
		return False

	x1, y1 = rect.x - rect1.x, rect.y - rect1.y
	x2, y2 = rect.x - rect2.x, rect.y - rect2.y

	for x in xrange(rect.width):
		for y in xrange(rect.height):
			if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
				return True
	return False

def get_hitmask(image):
	mask = []
	for x in range(image.get_width()):
		mask.append([])
		for y in range(image.get_height()):
			mask[x].append(bool(image.get_at((x,y))[3]))
	return mask

if __name__ == "__main__":
	main()
