from tqdm import tqdm

import torch

class Adversarial(object):
    def __init__(self, model, criterion, device, epsilon, iters, alpha):
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon
        self.iters = iters
        self.alpha = alpha
        self.device = device
        
    def attack(self, loader):
        correct = 0
        for image, label in tqdm(loader):
            # Send image and label to available hardware
            image = image.to(self.device)
            label = label.to(self.device)
            # Add adversarial noise to image
            adv_image = self.perturbe(image, label)
            # Classify the perturbed image (forward pass)
            output = self.model(adv_image)
            # Get adversarial prediction
            adv_pred = torch.max(output, 1)[1]
            # Update number of correct predictions
            correct += (adv_pred == label).sum().item()
        # Calculate total accuracy
        acc = correct / float(len(loader))
        return acc 
            
    def perturbe(self, image, label):
        pass
    
    
class FGSM(Adversarial):
    def __init__(self, model, criterion, device, epsilon=0.01, iters=5, alpha=0.008):
        super(FGSM, self).__init__(model, criterion, device, epsilon, iters, alpha)
        
    def perturbe(self, image, label):
        # Set as trainable
        image.requires_grad = True
        # Perform a forward pass
        output = self.model(image)
        # Get the loss for the original image
        loss = self.criterion(output, label).to(self.device)
        # Zero the gradients
        self.model.zero_grad()
        # Perform backpropagation
        loss.backward()
        # Get gradient sign
        grad_sign = image.grad.data.sign()
        # Create a perturbed image
        adv_image = torch.clamp(image + self.epsilon * grad_sign, 0, 1)
        
        return adv_image
    
    
class PGD(Adversarial):
    def __init__(self, model, criterion, device, epsilon=0.01, iters=5, alpha=0.008):
        super(PGD, self).__init__(model, criterion, device, epsilon, iters, alpha)
        
    def perturbe(self, image, label):
        # Save original image
        orig_image = image.data
        # Track gradients for the input image
        image.requires_grad = True
        # Iterate over a number of perturbations
        for _ in range(self.iters):
            # Each time forward the image
            # through the network layers
            output = self.model(image)
            # Zero model's existing gradients
            self.model.zero_grad()
            # Get the current iteration's loss
            loss = self.criterion(output, label).to(self.device)
            # Perform backpropagation retaining 
            # the previous graph values
            loss.backward(retain_graph=True)
            # Create a perturbation, similar to how FGSM works
            adv_image = image + self.alpha * image.grad.data.sign()
            # Calculate the difference (η) between adversarial 
            # and original image, while also bounding it between [-epsilon, epsilon] 
            # to keep the projections bounded 
            diff = torch.clamp(adv_image - orig_image, min=-self.epsilon, max=self.epsilon)
            # Add the difference with the original image to create a 
            # a new layer of perturbation and also bound the result between [0, 1]
            image = torch.clamp(orig_image + diff, min=0, max=1)
            # Tell pytorch to retain the previous gradients of the image,
            # otherwise they will automatically be deleted
            image.retain_grad()
            
        return image  