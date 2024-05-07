import torch
from torchvision import models


class FeatureReconstructionLoss:
    """
    Class for initializing VGG-19 model for feature reconstruction loss.
    """

    def __init__(self, w1 = 1, w2 = 1, w3 = 1) -> None:
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.mse_loss = torch.nn.MSELoss()
        self.model.to('cuda')
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        


    def get_intermediate_output(self, input_tensor):
        intermediate_output = None
        def hook(module, input, output):
            nonlocal intermediate_output
            intermediate_output = output


        layer_index = 26 # for the block5_conv4 convolution layer's ReLU output
        hook_handle = self.model.features[layer_index].register_forward_hook(hook)

        self.model(input_tensor)
        hook_handle.remove()
        return intermediate_output

    # Define a method to calculate the reconstruction loss for RGB channels
    def reconstruction_loss_rgb(self, im1, im2):
        original_features = self.get_intermediate_output(im1[:,:3])
        generated_features = self.get_intermediate_output(im2[:,:3])
        return torch.mean(torch.square(original_features-generated_features))


    # Define a method to calculate the reconstruction loss for depth channel
    def reconstruction_loss_depth(self, im1, im2):
        img1_depth = im1[:,3:]
        img2_depth = im2[:,3:]
        img1_depth = torch.cat([img1_depth,img1_depth,img1_depth], dim=1)
        img2_depth = torch.cat([img2_depth,img2_depth,img2_depth], dim=1)

        original_features = self.get_intermediate_output(img2_depth)
        generated_features = self.get_intermediate_output(img2_depth)
        return torch.mean(torch.square(original_features-generated_features))
    

    def combined_loss(self, im1, im2):
        return self.w1*self.reconstruction_loss_depth(im1, im2) + \
            self.w2*self.reconstruction_loss_rgb(im1, im2) + self.w3*(self.mse_loss(im1, im2))


if __name__ == '__main__':
    vgg19 = FeatureReconstructionLoss()
    input_tensor = torch.rand(1, 3, 224, 224)
    input_tensor2 = torch.rand(1, 3, 224, 224)
    
    print('putting tensor1 to cuda')
    input_tensor = input_tensor.cuda()
    
    print('putting tensor2 to cuda')
    input_tensor2 = input_tensor2.cuda()

    output = vgg19.reconstruction_loss_rgb(input_tensor, input_tensor2)
    print(vgg19.model)
    print(output)





