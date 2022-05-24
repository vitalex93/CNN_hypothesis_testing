from skimage.util import random_noise
from PIL import Image
import numpy as np
import random
import math


class NoNoise(object):
    def __init__(self, cmap="RGB") -> None:
        self.cmap = cmap
        
    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, value):
        if value != "RGB" and value != "L":
            raise ValueError("[!!] Random eraser only supports grayscale and RGB images --> provided value was {}".format(value))
        self._cmap = value
        
    def __call__(self, image):
        return image
    
    def __str__(self):
        return "Color-map: {}\n\n".format(self.cmap)
        

class GaussianNoise(NoNoise):
    def __init__(self, mean, std, cmap="RGB") -> None:
        super().__init__(cmap)
        self.mean = mean
        self.std = std
        
    @property
    def mean(self):
        return self._mean
    
    @mean.setter
    def mean(self, value):
        if value < 0:
            raise ValueError("[!!] WARN: The mean of the gaussian's noise distribution can't be negative.")
        self._mean = value
        
    @property
    def std(self):
        return self._std
    
    @std.setter
    def std(self, value):
        if value < 0:
            raise ValueError("[!!] WARN: The standard deviation of the gaussian's noise distribution can't be negative.")
        self._std = value
        
    def  __call__(self, image):
        new_image = np.array(image)
        # Apply gaussian noise to image
        new_image = random_noise(new_image, mode="gaussian", mean=self.mean, var=self.std)
        new_image *= 255
        # Cast back to PIL
        new_image = Image.fromarray(new_image.astype('uint8'), self.cmap)
        return new_image
    
    def __str__(self):
        return super(GaussianNoise, self).__str__() \
            + "Mean: {}\n\n".format(self.mean) \
            + "Standard deviation: {}\n\n".format(self.std)
    

class SpeckleNoise(GaussianNoise):
    def __init__(self, mean, std, cmap="RGB") -> None:
        super().__init__(mean, std, cmap) 
        
    def  __call__(self, image):
        new_image = np.array(image)
        new_image = random_noise(new_image, mode="speckle", mean=self.mean, var=self.std)
        new_image *= 255
        new_image = Image.fromarray(new_image.astype('uint8'), self.cmap)
        return new_image
    
    def __str__(self):
        return super(SpeckleNoise, self).__str__()
    
    
class SaltPepperNoise(NoNoise):
    def __init__(self, amount, cmap="RGB") -> None:
        super().__init__(cmap)
        self.amount = amount
        
    @property
    def amount(self):
        return self._amount
    
    @amount.setter
    def amount(self, value):
        if value < 0 or value > 1:
            raise ValueError("[!!] WARN: The amount of salt and pepper noise should range between [0-1]")
        self._amount = value
        
    def __call__(self, image):
        new_image = np.array(image)
        new_image = random_noise(new_image, mode="s&p", amount=self.amount)
        new_image *= 255
        new_image = Image.fromarray(new_image.astype('uint8'), self.cmap)
        return new_image
    
    def __str__(self):
        return super(SaltPepperNoise, self).__str__() + "Amount: {}\n\n".format(self.amount)
    
    
class PoissonNoise(NoNoise):
    def __init__(self, cmap="RGB") -> None:
        super().__init__(cmap)
        
    def __call__(self, image):
        new_image = np.array(image)
        new_image = random_noise(new_image, mode="poisson")
        new_image *= 255
        new_image = Image.fromarray(new_image.astype('uint8'), self.cmap)
        return new_image
    
    def __str__(self):
        return super(PoissonNoise, self).__str__()
    

class RandomEraser(NoNoise):
    """
    A class which applies different types of noise on an input image

    Attributes:
        subarea_low: the lower bound of the subarea size
        subarea_high: the upper bound of the subarea size
        aspect_ratio_low: the lower bound of the erasing aspect ratio
        aspect_ratio_high: the upper bound of the erasing aspect ratio
        init_erasing_prob: the initial probability of erasing a subarea
    """
    
    def __init__ (
        self, 
        init_erasing_prob=None,
        subarea_low=3,
        subarea_high=6,
        aspect_ratio_low=1,
        aspect_ratio_high=5,
        attempts=100,
        cmap="RGB"
    ) -> None:
        super().__init__(cmap)
        self.attempts = attempts
        self.subarea_low = subarea_low
        self.subarea_high = subarea_high
        self.aspect_ratio_low = aspect_ratio_low
        self.aspect_ratio_high = aspect_ratio_high
        self.init_erasing_prob = init_erasing_prob
        
    def __str__(self):
        return super(RandomEraser, self).__str__() \
            + "Initial erasing probability: {}\n\n".format(self.init_erasing_prob) \
            + "Subarea low bound: {}\n\n".format(self.subarea_low) \
            + "Subarea high bound: {}\n\n".format(self.subarea_high) \
            + "Aspect ratio low bound: {}\n\n".format(self.aspect_ratio_low) \
            + "Aspect ratio high bound: {}\n\n".format(self.aspect_ratio_high) \
            + "Erasing attempts: {}\n\n".format(self.attempts)
        
    @property
    def init_erasing_prob(self):
        return self._init_erasing_prob

    @init_erasing_prob.setter
    def init_erasing_prob(self, value):
        if value is None:
            self._init_erasing_prob = random.uniform(0, 1)
        elif value < 0 or value > 1:
            raise ValueError("[!!] WARN: Probabilities range from [0-1], provided value -> {}".format(value))
        else:
            self._init_erasing_prob = value

    @property
    def attempts(self):
        return self._attempts

    @attempts.setter
    def attempts(self, value):
        if value < 0:
            print("[!!] WARN: Invalid attempts number provided, applying default value = {}".format(100))
            self._attempts = 100
        else:
            self._attempts = value
            
    @property
    def subarea_low(self):
        return self._subarea_low

    @subarea_low.setter
    def subarea_low(self, value):
        if value <= 0:
            raise ValueError("[!!] WARN: Minimum subarea must be greater than zero")
        self._subarea_low = value

    @property
    def subarea_high(self):
        return self._subarea_high

    @subarea_high.setter
    def subarea_high(self, value):
        if value <= 0:
            raise ValueError("[!!] WARN: Maximum subarea must be greater than zero")
        self._subarea_high = value

    @property
    def aspect_ratio_low(self):
        return self._aspect_ratio_low

    @aspect_ratio_low.setter
    def aspect_ratio_low(self, value):
        if value <= 0:
            raise ValueError("[!!] WARN: Minimum aspect ratio must be greater than zero")
        self._aspect_ratio_low = value

    @property
    def aspect_ratio_high(self):
        return self._aspect_ratio_high

    @aspect_ratio_high.setter
    def aspect_ratio_high(self, value):
        if value <= 0:
            raise ValueError("[!!] WARN: Maximum aspect ratio must be greater than zero")
        self._aspect_ratio_high = value
        
    def __call__(self, image):
        """
        Apply noise to part of the image

        Args:
            image (ndarray): a numpy array representing the input image
            noise (str, optional): the noise type to add. Defaults to "s&p".

        Raises:
            ValueError: signifies an erroneous value being passed as argument

        Returns:
            ndarray: either returns the original image unchanged or an erased version of it
        """
        
        image = np.array(image)
        
        # Get the height (pixels in each column)
        # and the width (pixels in each row) of the image
        height, width, _ = image.shape
        total_area = height * width
        
        # Probability to decide upon erasing or not
        erasing_prob = random.uniform(0, 1)
        if erasing_prob >= self._init_erasing_prob:
            for _ in range(self.attempts):
                # Randomly create an erasing subarea
                divisor = random.randint(self.subarea_low, self.subarea_high)
                erasing_subarea = total_area / divisor
                erasing_subarea = int(erasing_subarea)
                
                # Randomly create an erasing aspect ratio
                erasing_aspect_ratio = random.uniform(self.aspect_ratio_low, self.aspect_ratio_high)
                
                # Calculate cliping window height and width
                height_clip = int(math.sqrt(erasing_subarea * erasing_aspect_ratio))
                width_clip = int(math.sqrt(erasing_subarea / erasing_aspect_ratio))
                
                # Get lower left corner of new clipping window
                start_x = random.randint(0, width)
                start_y = random.randint(0, height)
                
                # Get upper right corner of new clipping window
                border_y = start_y + height_clip
                border_x = start_x + width_clip
                
                # If the window fits inside the original, add noise to it
                if border_x <= width and border_y <= height:
                    mean = np.mean(image)
                    # Apply noise to the subwindow
                    for y in range(start_y, border_y - 1, 1):
                        for x in range(start_x, border_x - 1, 1):
                            image[y][x] = mean
                    break
                
        # Return original image
        image = Image.fromarray(image.astype('uint8'), self.cmap)
        return image