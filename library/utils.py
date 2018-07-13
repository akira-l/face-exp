# read parameter from file
import os
class ParameterReader(object):
	def __init__(self, _filename="parameters.txt"):
		self.filename = _filename
		fp = open(self.filename, 'r')
		self.parameters = {}
		while True:
			line = fp.readline() # '\n' is part of line
			if line=="": # line="" means end of file
				break
			line = line.strip()
			if line=="": # after strip, line=="": white line
				continue
			if (line[0]=='#'): # skip comment line
				continue
#			print("line=|"+line+"|")
			pair = line.split('=')
			if len(pair)!=2:
				continue
			key = pair[0].strip()
			value = pair[1].strip()
			self.parameters[key] = value
		fp.close()
	def getData(self, key):
		try:
			ret = self.parameters[key.strip()]
		except:
			print("EXCEPTION:no parameter called:%s!!!"%key)
		return ret

from PIL import Image
import numpy as np
from skimage.transform import warp, AffineTransform

class RandomAffineTransform(object):
	def __init__(self,
				scale_range,
				rotation_range,
				shear_range,
				translation_range
				):
		self.scale_range = scale_range
		self.rotation_range = rotation_range
		self.shear_range = shear_range
		self.translation_range = translation_range

	def __call__(self, img):
		img_data = np.array(img)
		h, w, n_chan = img_data.shape
		scale_x = np.random.uniform(*self.scale_range)
		scale_y = np.random.uniform(*self.scale_range)
		scale = (scale_x, scale_y)
		rotation = np.random.uniform(*self.rotation_range)
		shear = np.random.uniform(*self.shear_range)
		translation = (
			np.random.uniform(*self.translation_range) * w,
			np.random.uniform(*self.translation_range) * h
		)
		af = AffineTransform(scale=scale, shear=shear, rotation=rotation, translation=translation)
		img_data1 = warp(img_data, af.inverse)
		img1 = Image.fromarray(np.uint8(img_data1 * 255))
		return img1
def prepareSaveDir(rootSaveDir, cameraIP):
	ipClassDir = os.path.join(rootSaveDir, "readCamera", cameraIP)
	if not os.path.isdir(ipClassDir):
		os.makedirs(ipClassDir)
