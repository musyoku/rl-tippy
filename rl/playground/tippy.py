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

ACTION_NO_OP = 0
ACTION_JUMP = 1

class TippyAgent(object):
	def __init__(self, train=True):
		self.fps = 30
		self.train = train
		self.screenwidth  = 288
		self.screenheight = 512
		self.pipegapsize  = 100 # gap between upper and lower part of pipe
		self.basey        = self.screenheight * 0.79
		self.images = {}
		self.sounds = {}
		self.hitmasks = {}

		self.player_images = (
			"assets/sprites/tippy-upflap.png",
			"assets/sprites/tippy-midflap.png",
		)

		self.background_image = "assets/sprites/background.png"
		self.pipe_image = "assets/sprites/pipe.png"

		self.screen = None
		self.fpsclock = None

		self.action = ACTION_NO_OP

	def play(self):
		pygame.init()
		self.fpsclock = pygame.time.Clock()
		self.screen = pygame.display.set_mode((self.screenwidth, self.screenheight))
		pygame.display.set_caption("Flappy Tippy")

		# numbers sprites for score display
		self.images["numbers"] = (
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
		self.images["gameover"] = pygame.image.load("assets/sprites/gameover.png").convert_alpha()
		# message sprite for welcome screen
		self.images["message"] = pygame.image.load("assets/sprites/message.png").convert_alpha()
		# base (ground) sprite
		self.images["base"] = pygame.image.load("assets/sprites/base.png").convert_alpha()

		# sounds
		if "win" in sys.platform:
			soundExt = ".wav"
		else:
			soundExt = ".ogg"

		self.sounds["die"]    = pygame.mixer.Sound("assets/audio/die" + soundExt)
		self.sounds["hit"]    = pygame.mixer.Sound("assets/audio/hit" + soundExt)
		self.sounds["point"]  = pygame.mixer.Sound("assets/audio/point" + soundExt)
		self.sounds["swoosh"] = pygame.mixer.Sound("assets/audio/swoosh" + soundExt)
		self.sounds["wing"]   = pygame.mixer.Sound("assets/audio/wing" + soundExt)

		self.images["background"] = pygame.image.load(self.background_image).convert()

		self.images["player"] = (
			pygame.image.load(self.player_images[0]).convert_alpha(),
			pygame.image.load(self.player_images[1]).convert_alpha(),
		)

		self.images["pipe"] = (
			pygame.transform.rotate(pygame.image.load(self.pipe_image).convert_alpha(), 180),
			pygame.image.load(self.pipe_image).convert_alpha(),
		)

		# hismask for pipes
		self.hitmasks["pipe"] = (
			self.get_hitmask(self.images["pipe"][0]),
			self.get_hitmask(self.images["pipe"][1]),
		)

		# hitmask for player
		self.hitmasks["player"] = (
			self.get_hitmask(self.images["player"][0]),
			self.get_hitmask(self.images["player"][1]),
			# self.get_hitmask(self.images["player"][2]),
		)
		while True:
			movementInfo = self.show_welcome_animation()
			crashInfo = self.run_main_game(movementInfo)
			self.show_game_over_screen(crashInfo)

	def show_welcome_animation(self):
		"""Shows welcome screen animation of flappy bird"""
		# index of player to blit on screen
		playerIndex = 0
		playerIndexGen = cycle([0, 1])
		# iterator used to change playerIndex after every 5th iteration
		loopIter = 0

		playerx = int(self.screenwidth * 0.2)
		playery = int((self.screenheight - self.images["player"][0].get_height()) / 2)

		messagex = int((self.screenwidth - self.images["message"].get_width()) / 2)
		messagey = int(self.screenheight * 0.12)

		basex = 0
		# amount by which base can maximum shift to left
		baseShift = self.images["base"].get_width() - self.images["background"].get_width()

		# player shm for up-down motion on welcome screen
		playerShmVals = {"val": 0, "dir": 1}

		while True:
			for event in pygame.event.get():
				if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
					pygame.quit()
					sys.exit()
				if (event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP)):
					# make first flap sound and return values for run_main_game
					self.sounds["wing"].play()
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
			self.player_shm(playerShmVals)

			# draw sprites
			self.screen.blit(self.images["background"], (0,0))
			self.screen.blit(self.images["player"][playerIndex], (playerx, playery + playerShmVals["val"]))
			self.screen.blit(self.images["message"], (messagex, messagey))
			self.screen.blit(self.images["base"], (basex, self.basey))

			pygame.display.update()
			self.fpsclock.tick(self.fps)

	def run_main_game(self, movementInfo):
		score = playerIndex = loopIter = 0
		playerIndexGen = movementInfo["playerIndexGen"]
		playerx, playery = int(self.screenwidth * 0.2), movementInfo["playery"]

		basex = movementInfo["basex"]
		baseShift = self.images["base"].get_width() - self.images["background"].get_width()

		# get 2 new pipes to add to upperPipes lowerPipes list
		newPipe1 = self.get_random_pipe()
		newPipe2 = self.get_random_pipe()

		# list of upper pipes
		upperPipes = [
			{"x": self.screenwidth + 200, "y": newPipe1[0]["y"]},
			{"x": self.screenwidth + 200 + (self.screenwidth / 2), "y": newPipe2[0]["y"]},
		]

		# list of lowerpipe
		lowerPipes = [
			{"x": self.screenwidth + 200, "y": newPipe1[1]["y"]},
			{"x": self.screenwidth + 200 + (self.screenwidth / 2), "y": newPipe2[1]["y"]},
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
			rl_action = self.action
			rl_reward = 0
			for event in pygame.event.get():
				if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
					pygame.quit()
					sys.exit()
				if (event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP)) or self.action == ACTION_JUMP:
					if playery > -2 * self.images["player"][0].get_height():
						playerVelY = playerFlapAcc
						playerFlapped = True
						self.sounds["wing"].play()

			# check for crash here
			crashTest = self.check_crash({"x": playerx, "y": playery, "index": playerIndex}, upperPipes, lowerPipes)
			if crashTest[0]:
				rl_reward = -1
				if self.train == False:
					self.agent_end(rl_action, rl_reward)
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
			playerMidPos = playerx + self.images["player"][0].get_width() / 2
			for pipe in upperPipes:
				pipeMidPos = pipe["x"] + self.images["pipe"][0].get_width() / 2
				if pipeMidPos <= playerMidPos < pipeMidPos + 4:
					score += 1
					self.sounds["point"].play()
					rl_reward = 1

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
			playerHeight = self.images["player"][playerIndex].get_height()
			playery += min(playerVelY, self.basey - playery - playerHeight)

			# move pipes to left
			for uPipe, lPipe in zip(upperPipes, lowerPipes):
				uPipe["x"] += pipeVelX
				lPipe["x"] += pipeVelX

			# add new pipe when first pipe is about to touch left of screen
			if 0 < upperPipes[0]["x"] < 5:
				newPipe = self.get_random_pipe()
				upperPipes.append(newPipe[0])
				lowerPipes.append(newPipe[1])

			# remove first pipe if its out of the screen
			if upperPipes[0]["x"] < -self.images["pipe"][0].get_width():
				upperPipes.pop(0)
				lowerPipes.pop(0)

			# draw sprites
			self.screen.blit(self.images["background"], (0,0))

			for uPipe, lPipe in zip(upperPipes, lowerPipes):
				self.screen.blit(self.images["pipe"][0], (uPipe["x"], uPipe["y"]))
				self.screen.blit(self.images["pipe"][1], (lPipe["x"], lPipe["y"]))

			self.screen.blit(self.images["base"], (basex, self.basey))
			# print score so player overlaps the score
			self.show_score(score)
			self.screen.blit(self.images["player"][playerIndex], (playerx, playery))

			pygame.display.update()
			self.fpsclock.tick(self.fps)

			# capture screen
			buf = self.screen.get_buffer()
			image = Image.frombytes("RGBA",self.screen.get_size(),buf.raw)
			del buf
			rl_next_frame = np.asarray(image, dtype=np.uint8)
			rl_next_frame = 0.2126 * rl_next_frame[..., 2] + 0.7152 * rl_next_frame[..., 1] + 0.0722 * rl_next_frame[..., 0]
			rl_next_frame = scipy.misc.imresize(rl_next_frame, size=(96, 72), interp="bilinear")
			rl_next_frame = rl_next_frame[0:72, 0:72]
			# Image.fromarray(rl_next_frame).convert("RGB").save("screen.bmp")

			self.agent_step(rl_action, rl_reward, rl_next_frame, score)

	def agent_step(self, action, reward, next_frame, score):
		raise NotImplementedError()

	def agent_end(self, action, reward, score):
		raise NotImplementedError()

	def show_game_over_screen(self, crashInfo):
		"""crashes the player down ans shows gameover image"""
		score = crashInfo["score"]
		playerx = self.screenwidth * 0.2
		playery = crashInfo["y"]
		playerHeight = self.images["player"][0].get_height()
		playerVelY = crashInfo["playerVelY"]
		playerAccY = 2

		basex = crashInfo["basex"]

		upperPipes, lowerPipes = crashInfo["upperPipes"], crashInfo["lowerPipes"]

		# play hit and die sounds
		self.sounds["hit"].play()
		if not crashInfo["groundCrash"]:
			self.sounds["die"].play()

		while True:
			for event in pygame.event.get():
				if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
					pygame.quit()
					sys.exit()
				if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
					if playery + playerHeight >= self.basey - 1:
						return

			# player y shift
			if playery + playerHeight < self.basey - 1:
				playery += min(playerVelY, self.basey - playery - playerHeight)

			# player velocity change
			if playerVelY < 15:
				playerVelY += playerAccY

			# draw sprites
			self.screen.blit(self.images["background"], (0,0))

			for uPipe, lPipe in zip(upperPipes, lowerPipes):
				self.screen.blit(self.images["pipe"][0], (uPipe["x"], uPipe["y"]))
				self.screen.blit(self.images["pipe"][1], (lPipe["x"], lPipe["y"]))

			self.screen.blit(self.images["base"], (basex, self.basey))
			self.show_score(score)
			self.screen.blit(self.images["player"][1], (playerx,playery))

			self.fpsclock.tick(self.fps)
			pygame.display.update()

	def player_shm(self, playerShm):
		"""oscillates the value of playerShm["val"] between 8 and -8"""
		if abs(playerShm["val"]) == 8:
			playerShm["dir"] *= -1

		if playerShm["dir"] == 1:
			 playerShm["val"] += 1
		else:
			playerShm["val"] -= 1

	def get_random_pipe(self):
		"""returns a randomly generated pipe"""
		# y of gap between upper and lower pipe
		gapY = random.randrange(0, int(self.basey * 0.6 - self.pipegapsize))
		gapY += int(self.basey * 0.2)
		pipeHeight = self.images["pipe"][0].get_height()
		pipeX = self.screenwidth + 10

		return [
			{"x": pipeX, "y": gapY - pipeHeight},  # upper pipe
			{"x": pipeX, "y": gapY + self.pipegapsize}, # lower pipe
		]

	def show_score(self, score):
		"""displays score in center of screen"""
		scoreDigits = [int(x) for x in list(str(score))]
		totalWidth = 0 # total width of all numbers to be printed

		for digit in scoreDigits:
			totalWidth += self.images["numbers"][digit].get_width()

		Xoffset = (self.screenwidth - totalWidth) / 2

		for digit in scoreDigits:
			self.screen.blit(self.images["numbers"][digit], (Xoffset, self.screenheight * 0.1))
			Xoffset += self.images["numbers"][digit].get_width()

	def check_crash(self, player, upperPipes, lowerPipes):
		"""returns True if player collders with base or pipes."""
		pi = player["index"]
		player["w"] = self.images["player"][0].get_width()
		player["h"] = self.images["player"][0].get_height()

		# if player crashes into ground
		if player["y"] + player["h"] >= self.basey - 1:
			return [True, True]
		else:

			playerRect = pygame.Rect(player["x"], player["y"], player["w"], player["h"])
			pipeW = self.images["pipe"][0].get_width()
			pipeH = self.images["pipe"][0].get_height()

			for uPipe, lPipe in zip(upperPipes, lowerPipes):
				# upper and lower pipe rects
				uPipeRect = pygame.Rect(uPipe["x"], uPipe["y"], pipeW, pipeH)
				lPipeRect = pygame.Rect(lPipe["x"], lPipe["y"], pipeW, pipeH)

				# player and upper/lower pipe hitmasks
				pHitMask = self.hitmasks["player"][pi]
				uHitmask = self.hitmasks["pipe"][0]
				lHitmask = self.hitmasks["pipe"][1]

				# if bird collided with upipe or lpipe
				uCollide = self.pixel_collision(playerRect, uPipeRect, pHitMask, uHitmask)
				lCollide = self.pixel_collision(playerRect, lPipeRect, pHitMask, lHitmask)

				if uCollide or lCollide:
					return [True, False]

		return [False, False]

	def pixel_collision(self, rect1, rect2, hitmask1, hitmask2):
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

	def get_hitmask(self, image):
		mask = []
		for x in range(image.get_width()):
			mask.append([])
			for y in range(image.get_height()):
				mask[x].append(bool(image.get_at((x,y))[3]))
		return mask

if __name__ == "__main__":
	tippy = Tippy()
	tippy.play()