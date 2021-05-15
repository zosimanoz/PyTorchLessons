
# create tensors in pytorch
import torch
import matplotlib.pyplot as plt
import numpy as np

x_tensor = torch.tensor([1,2,3], dtype=torch.float32)
print(x_tensor)

x_tensor = torch.tensor([[1,2, 3], [9,8,7]], device="cpu")
print(x_tensor)
print(x_tensor.dtype)
print(x_tensor.device)

y_tensor = torch.tensor([[3,4,5], [5,7,8]])
print(y_tensor)
z_tensor_add_xy = torch.add(x_tensor, y_tensor)
print(z_tensor_add_xy)

z_tensor_add = x_tensor + y_tensor
print(z_tensor_add)

z_tensor_mul = torch.mul(x_tensor, y_tensor)
print(f'multiply matrixes (mul function multiply elements of matrixes): {z_tensor_mul}')


new_x_tensor = torch.tensor([[1,2,3], [4,5,6]])
new_y_tensor = torch.tensor([[1,2], [3,4], [5, 6]])
print(f'shape of matrix X : {new_x_tensor.shape}, Y : {new_y_tensor.shape}')
print('shapes should match mxn and nxm for matrix multiplications and resulting shape is mxm')
z_tensor_mm = torch.mm(new_x_tensor, new_y_tensor)
print(f'matrix multiply mm (actual matrix multiplication shape should match): {z_tensor_mm}')


#######################
#######################
#######################
''' Tensor indexing '''
batch_size = 32
features = 25
input_x_tensor = torch.rand((batch_size, features))
print(f'shape of input tensor: {input_x_tensor.shape} and values: {input_x_tensor}')

print(f'first batch: {input_x_tensor[0]}')

# get fifth batch and only element from index 1 to 5
print(input_x_tensor[5, 1:5])


x = torch.arange(10)
print(x)
# get all elements greater than 2 from matrix
print(x[x > 2])
# get all element greater than 2 and less than 8
print(x[(x>2) & (x<8)])
# get all element greater than 2 or less than 8
print(x[(x > 2) | (x < 2)])


###############
###############
###############
''' Tensor reshaping in pytorch '''
x = torch.arange(9)
x_3x3 = x.view(3,3)
print(x_3x3)

x_3x3_reshape = x.reshape(3,3)
print(x_3x3_reshape)


###############################
###############################
###############################
''' Creating Images using Pytorch tensor '''

x_img = torch.rand(64)
print(x_img.shape)

x_image_reshaped = x_img.reshape(8,8)
print(x_image_reshaped.shape)

plt.imshow(x_image_reshaped, cmap='gray')
plt.show()

''' Make 3 channel image '''
channels = 3
image_size = 1728

x_rgb_img = torch.randint(0, 255, (1728, 1))
print(x_rgb_img.shape)

x_rgb_img = torch.ravel(x_rgb_img).reshape(3, 24, 24)
print(x_rgb_img.shape)
x_rgb_img = x_rgb_img.permute(2,1, 0)
print(x_rgb_img.shape)
# matplot lib cannot display image as channel first
# so we need to reshape the image to channel at end

plt.imshow(x_rgb_img)
plt.show()


# using numpy
x = np.random.random((10, 10, 3))
print(f'numpy array x shape: {x.shape}')
print(x)
plt.imshow(x)
plt.show()


# custom logic to move the axis of images
x_img_new = np.zeros((3, 10, 10))
print(f'img new shape: {x_img_new.shape}')
for i in range(3):
    x_img_new[i, :, :] = x[:, :, i]

print(f'custom loop reshape : {x_img_new}')



x = np.moveaxis(x, -1,0)
print(f'numpy array x shape: {x.shape}')
print(x)
plt.imshow(x[0, :, :])
plt.show()