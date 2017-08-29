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

OBSERVATION_WIDTH = 72
OBSERVATION_HEIGHT = 72

class TippyAgent(object):
	def __init__(self, lives=5):
		self._fps = 30
		self._lives = lives
		self._screenwidth  = 288
		self._screenheight = 512
		self._pipegapsize  = 100 # gap between upper and lower part of pipe
		self._basey        = self._screenheight * 0.79
		self._images = {}
		self._sounds = {}
		self._hitmasks = {}

		self._player_images = (
			"../../../rl/playground/assets/sprites/tippy-upflap.png",
			"../../../rl/playground/assets/sprites/tippy-midflap.png",
		)

		self._background_image = "../../../rl/playground/assets/sprites/background.png"
		self._pipe_image = "../../../rl/playground/assets/sprites/pipe.png"

		self._screen = None
		self._fpsclock = None
		self._rl_prev_frame = None

	def set_pipegapsize(self, size):
		self._pipegapsize = size

	def set_lives(self, lives):
		self._lives = lives

	def play(self):
		pygame.init()
		self._fpsclock = pygame.time.Clock()
		self._screen = pygame.display.set_mode((self._screenwidth, self._screenheight))
		pygame.display.set_caption("Flappy Tippy")

		# numbers sprites for score display
		self._images["numbers"] = (
			pygame.image.load("../../../rl/playground/assets/sprites/0.png").convert_alpha(),
			pygame.image.load("../../../rl/playground/assets/sprites/1.png").convert_alpha(),
			pygame.image.load("../../../rl/playground/assets/sprites/2.png").convert_alpha(),
			pygame.image.load("../../../rl/playground/assets/sprites/3.png").convert_alpha(),
			pygame.image.load("../../../rl/playground/assets/sprites/4.png").convert_alpha(),
			pygame.image.load("../../../rl/playground/assets/sprites/5.png").convert_alpha(),
			pygame.image.load("../../../rl/playground/assets/sprites/6.png").convert_alpha(),
			pygame.image.load("../../../rl/playground/assets/sprites/7.png").convert_alpha(),
			pygame.image.load("../../../rl/playground/assets/sprites/8.png").convert_alpha(),
			pygame.image.load("../../../rl/playground/assets/sprites/9.png").convert_alpha()
		)

		# game over sprite
		self._images["gameover"] = pygame.image.load("../../../rl/playground/assets/sprites/gameover.png").convert_alpha()
		# message sprite for welcome screen
		self._images["message"] = pygame.image.load("../../../rl/playground/assets/sprites/message.png").convert_alpha()
		# base (ground) sprite
		self._images["base"] = pygame.image.load("../../../rl/playground/assets/sprites/base.png").convert_alpha()

		# sounds
		if "win" in sys.platform:
			soundExt = ".wav"
		else:
			soundExt = ".ogg"

		self._sounds["die"]    = pygame.mixer.Sound("../../../rl/playground/assets/audio/die" + soundExt)
		self._sounds["hit"]    = pygame.mixer.Sound("../../../rl/playground/assets/audio/hit" + soundExt)
		self._sounds["point"]  = pygame.mixer.Sound("../../../rl/playground/assets/audio/point" + soundExt)
		self._sounds["swoosh"] = pygame.mixer.Sound("../../../rl/playground/assets/audio/swoosh" + soundExt)
		self._sounds["wing"]   = pygame.mixer.Sound("../../../rl/playground/assets/audio/wing" + soundExt)

		self._images["background"] = pygame.image.load(self._background_image).convert()

		self._images["player"] = (
			pygame.image.load(self._player_images[0]).convert_alpha(),
			pygame.image.load(self._player_images[1]).convert_alpha(),
		)

		self._images["pipe"] = (
			pygame.transform.rotate(pygame.image.load(self._pipe_image).convert_alpha(), 180),
			pygame.image.load(self._pipe_image).convert_alpha(),
		)

		# hismask for pipes
		self._hitmasks["pipe"] = (
			self.get_hitmask(self._images["pipe"][0]),
			self.get_hitmask(self._images["pipe"][1]),
		)

		# hitmask for player
		self._hitmasks["player"] = (
			self.get_hitmask(self._images["player"][0]),
			self.get_hitmask(self._images["player"][1]),
			# self.get_hitmask(self._images["player"][2]),
		)

		playerShmVals = {"val": 0, "dir": 1}
		playerIndexGen = cycle([0, 1])
		playery = int((self._screenheight - self._images["player"][0].get_height()) / 2)
		movementInfo = {
			"playery": playery + playerShmVals["val"],
			"basex": 0,
			"playerIndexGen": playerIndexGen,
		}
		while True:
			crashInfo = self.run_main_game(movementInfo)

		# while True:
		# 	movementInfo = self.show_welcome_animation()
		# 	crashInfo = self.run_main_game(movementInfo)
		# 	self.show_game_over_screen(crashInfo)

	def capture_screen(self):
		buf = self._screen.get_buffer()
		image = Image.frombytes("RGBA",self._screen.get_size(),buf.raw)
		del buf
		frame = np.asarray(image, dtype=np.uint8)
		frame = 0.2126 * frame[..., 2] + 0.7152 * frame[..., 1] + 0.0722 * frame[..., 0]
		frame = scipy.misc.imresize(frame, size=(96, 72), interp="bilinear")
		frame = frame[0:72, 0:72]
		frame = frame.astype(np.float32) / 255
		frame = (frame - 0.5) * 2
		return frame

	def show_welcome_animation(self):
		"""Shows welcome screen animation of flappy bird"""
		# index of player to blit on screen
		playerIndex = 0
		playerIndexGen = cycle([0, 1])
		# iterator used to change playerIndex after every 5th iteration
		loopIter = 0

		playerx = int(self._screenwidth * 0.2)
		playery = int((self._screenheight - self._images["player"][0].get_height()) / 2)

		messagex = int((self._screenwidth - self._images["message"].get_width()) / 2)
		messagey = int(self._screenheight * 0.12)

		basex = 0
		# amount by which base can maximum shift to left
		baseShift = self._images["base"].get_width() - self._images["background"].get_width()

		# player shm for up-down motion on welcome screen
		playerShmVals = {"val": 0, "dir": 1}

		while True:
			for event in pygame.event.get():
				if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
					pygame.quit()
					sys.exit()
				if (event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP)):
					# make first flap sound and return values for run_main_game
					self._sounds["wing"].play()
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
			self._player_shm(playerShmVals)

			# draw sprites
			self._screen.blit(self._images["background"], (0,0))
			self._screen.blit(self._images["player"][playerIndex], (playerx, playery + playerShmVals["val"]))
			self._screen.blit(self._images["message"], (messagex, messagey))
			self._screen.blit(self._images["base"], (basex, self._basey))

			pygame.display.update()
			self._fpsclock.tick(self._fps)

	def run_main_game(self, movementInfo):
		score = playerIndex = loopIter = 0
		playerIndexGen = movementInfo["playerIndexGen"]
		playerx, playery = int(self._screenwidth * 0.2), movementInfo["playery"]

		basex = movementInfo["basex"]
		baseShift = self._images["base"].get_width() - self._images["background"].get_width()

		# get 2 new pipes to add to upperPipes lowerPipes list
		newPipe1 = self.get_random_pipe()
		newPipe2 = self.get_random_pipe()

		# list of upper pipes
		upperPipes = [
			{"x": self._screenwidth + 200, "y": newPipe1[0]["y"]},
			{"x": self._screenwidth + 200 + (self._screenwidth / 2), "y": newPipe2[0]["y"]},
		]

		# list of lowerpipe
		lowerPipes = [
			{"x": self._screenwidth + 200, "y": newPipe1[1]["y"]},
			{"x": self._screenwidth + 200 + (self._screenwidth / 2), "y": newPipe2[1]["y"]},
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
			rl_action = self.agent_action()
			rl_reward = 0
			for event in pygame.event.get():
				if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
					pygame.quit()
					sys.exit()
				if (event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP)):
					if playery > -2 * self._images["player"][0].get_height():
						playerVelY = playerFlapAcc
						playerFlapped = True
						# self._sounds["wing"].play()

			if rl_action == ACTION_JUMP:
				rl_reward = -0.1
				if playery > -2 * self._images["player"][0].get_height():
					playerVelY = playerFlapAcc
					playerFlapped = True
					self._sounds["wing"].play()

			# check for crash here
			crashTest = self.check_crash({"x": playerx, "y": playery, "index": playerIndex}, upperPipes, lowerPipes)
			if crashTest[0]:
				rl_reward = -1
				self._lives -= 1
				if self._lives < 1:
					assert self._rl_prev_frame is not None
					if self._rl_prev_frame is not None:
						self.agent_end(self._rl_prev_frame, rl_action, rl_reward, score)
					self._sounds["die"].play()
					return {
						"y": playery,
						"groundCrash": crashTest[1],
						"basex": basex,
						"upperPipes": upperPipes,
						"lowerPipes": lowerPipes,
						"score": score,
						"playerVelY": playerVelY,
					}
			else:
				# check for score
				playerMidPos = playerx + self._images["player"][0].get_width() / 2
				for pipe in upperPipes:
					pipeMidPos = pipe["x"] + self._images["pipe"][0].get_width() / 2
					if pipeMidPos <= playerMidPos < pipeMidPos + 4:
						score += 1
						self._sounds["point"].play()
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
			playerHeight = self._images["player"][playerIndex].get_height()
			playery += min(playerVelY, self._basey - playery - playerHeight)

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
			if upperPipes[0]["x"] < -self._images["pipe"][0].get_width():
				upperPipes.pop(0)
				lowerPipes.pop(0)

			# draw sprites
			self._screen.blit(self._images["background"], (0,0))

			for uPipe, lPipe in zip(upperPipes, lowerPipes):
				self._screen.blit(self._images["pipe"][0], (uPipe["x"], uPipe["y"]))
				self._screen.blit(self._images["pipe"][1], (lPipe["x"], lPipe["y"]))

			self._screen.blit(self._images["base"], (basex, self._basey))
			# print score so player overlaps the score
			self.show_score(score)
			self._screen.blit(self._images["player"][playerIndex], (playerx, playery))

			pygame.display.update()
			self._fpsclock.tick(self._fps)

			# capture screen
			rl_next_frame = self.capture_screen()
			# Image.fromarray(rl_next_frame).convert("RGB").save("screen.bmp")

			if self._rl_prev_frame is not None:
				self.agent_observe(self._rl_prev_frame, rl_action, rl_reward, rl_next_frame, score, self._lives)

			self._rl_prev_frame = rl_next_frame

	def agent_start(self):
		raise NotImplementedError()

	def agent_observe(self, action, reward, next_frame, score):
		raise NotImplementedError()

	def agent_end(self, action, reward, score):
		raise NotImplementedError()

	def show_game_over_screen(self, crashInfo):
		"""crashes the player down ans shows gameover image"""
		score = crashInfo["score"]
		playerx = self._screenwidth * 0.2
		playery = crashInfo["y"]
		playerHeight = self._images["player"][0].get_height()
		playerVelY = crashInfo["playerVelY"]
		playerAccY = 2

		basex = crashInfo["basex"]

		upperPipes, lowerPipes = crashInfo["upperPipes"], crashInfo["lowerPipes"]

		# play hit and die sounds
		self._sounds["hit"].play()
		if not crashInfo["groundCrash"]:
			self._sounds["die"].play()

		while True:
			for event in pygame.event.get():
				if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
					pygame.quit()
					sys.exit()
				if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
					if playery + playerHeight >= self._basey - 1:
						return

			# player y shift
			if playery + playerHeight < self._basey - 1:
				playery += min(playerVelY, self._basey - playery - playerHeight)

			# player velocity change
			if playerVelY < 15:
				playerVelY += playerAccY

			# draw sprites
			self._screen.blit(self._images["background"], (0,0))

			for uPipe, lPipe in zip(upperPipes, lowerPipes):
				self._screen.blit(self._images["pipe"][0], (uPipe["x"], uPipe["y"]))
				self._screen.blit(self._images["pipe"][1], (lPipe["x"], lPipe["y"]))

			self._screen.blit(self._images["base"], (basex, self._basey))
			self.show_score(score)
			self._screen.blit(self._images["player"][1], (playerx,playery))

			self._fpsclock.tick(self._fps)
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
		gapY = random.randrange(0, int(self._basey * 0.6 - self._pipegapsize))
		gapY += int(self._basey * 0.2)
		pipeHeight = self._images["pipe"][0].get_height()
		pipeX = self._screenwidth + 10

		return [
			{"x": pipeX, "y": gapY - pipeHeight},  # upper pipe
			{"x": pipeX, "y": gapY + self._pipegapsize}, # lower pipe
		]

	def show_score(self, score):
		"""displays score in center of screen"""
		scoreDigits = [int(x) for x in list(str(score))]
		totalWidth = 0 # total width of all numbers to be printed

		for digit in scoreDigits:
			totalWidth += self._images["numbers"][digit].get_width()

		Xoffset = (self._screenwidth - totalWidth) / 2

		for digit in scoreDigits:
			self._screen.blit(self._images["numbers"][digit], (Xoffset, self._screenheight * 0.1))
			Xoffset += self._images["numbers"][digit].get_width()

	def check_crash(self, player, upperPipes, lowerPipes):
		"""returns True if player collders with base or pipes."""
		pi = player["index"]
		player["w"] = self._images["player"][0].get_width()
		player["h"] = self._images["player"][0].get_height()

		# if player crashes into ground
		if player["y"] + player["h"] >= self._basey - 1:
			return [True, True]
		elif player["y"] + player["h"] <= 10:
			return [True, True]
		else:
			playerRect = pygame.Rect(player["x"], player["y"], player["w"], player["h"])
			pipeW = self._images["pipe"][0].get_width()
			pipeH = self._images["pipe"][0].get_height()

			for uPipe, lPipe in zip(upperPipes, lowerPipes):
				# upper and lower pipe rects
				uPipeRect = pygame.Rect(uPipe["x"], uPipe["y"], pipeW, pipeH)
				lPipeRect = pygame.Rect(lPipe["x"], lPipe["y"], pipeW, pipeH)

				# player and upper/lower pipe hitmasks
				pHitMask = self._hitmasks["player"][pi]
				uHitmask = self._hitmasks["pipe"][0]
				lHitmask = self._hitmasks["pipe"][1]

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