# import os
# import torch
from model_maker import get_network

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = get_network(patch_size = [168, 168, 16], spacing = [4.07, 4.07, 3.00])
# model = model.to(device)

# import torch
# from torchviz import make_dot


# # Create a dummy input tensor based on the patch_size
# dummy_input = torch.randn(1, 1, *[168, 168, 16]).to(device)

# # Perform a forward pass to get the output
# # The deep_supervision flag in get_network is set to True, the output will be a list
# output = model(dummy_input)[0] if isinstance(model(dummy_input), list) else model(dummy_input)

# # Visualize the graph
# dot = make_dot(output, params=dict(model.named_parameters()))

# # Save the visualization to a file
# dot.format = 'png'
# dot.render('dynunet_architecture')

# # Print the path to the saved visualization file
# print('The visualization of DynUNet is saved to "dynunet_architecture.png"')


# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Function to create a rectangle with text inside
# def draw_block(ax, position, text, color):
#     rect = patches.Rectangle(position, 0.1, 0.2, linewidth=1, edgecolor=color, facecolor='none')
#     ax.add_patch(rect)
#     ax.text(position[0] + 0.05, position[1] + 0.1, text, color=color, weight='bold', 
#             fontsize=12, ha='center', va='center')

# # Define figure and axis
# fig, ax = plt.subplots(figsize=(10, 3))
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 0.4)
# ax.axis('off')  # Turn off axes

# # Define positions of the blocks
# positions = [(0.1 * i, 0.1) for i in range(1, 10)]

# # Draw blocks for input, downsampling, bottleneck, upsampling, and output
# draw_block(ax, positions[0], 'Input\n[1x168x168x16]', 'green')
# for i, pos in enumerate(positions[1:4]):
#     draw_block(ax, pos, f'Down\n[{2**(i+1)}x{168//(2**(i+1))}x...]', 'blue')
# draw_block(ax, positions[4], 'Bottleneck\n[8x...x...]', 'red')
# for i, pos in enumerate(positions[5:8]):
#     draw_block(ax, pos, f'Up\n[{2**(3-i)}x{168//(2**(3-i))}x...]', 'orange')
# draw_block(ax, positions[8], 'Output\n[1x168x168x16]', 'purple')

# # Draw lines for skip connections
# for i in range(1, 4):
#     ax.annotate('', xy=positions[4], xycoords='data',
#                 xytext=positions[i], textcoords='data',
#                 arrowprops=dict(arrowstyle="<->", color="grey"))
    
# # Draw deep supervision connections
# ax.annotate('Deep\nSupervision', xy=(positions[7][0], positions[7][1] + 0.15), xycoords='data',
#             xytext=(positions[8][0], positions[8][1] + 0.25), textcoords='data',
#             arrowprops=dict(arrowstyle="->", color="grey"), ha='center')
# ax.annotate('', xy=(positions[6][0], positions[6][1] + 0.15), xycoords='data',
#             xytext=(positions[8][0], positions[8][1] + 0.25), textcoords='data',
#             arrowprops=dict(arrowstyle="->", color="grey"), ha='center')

# # Show the plot
# plt.tight_layout()
# plt.show()


# from torchsummary import summary

# model = get_network(patch_size=[168, 168, 16], spacing=[4.07, 4.07, 3.00])
# summary(model, input_size=(1, 168, 168, 16))  # Adjust the input_size as per your model

# from torchview import draw_graph

# model = MLP()
# batch_size = 2
# # device='meta' -> no memory is consumed for visualization
# model_graph = draw_graph(model, input_size=(batch_size, 128), device='meta')
# model_graph.visual_graph


# from torchview import draw_graph
# import torch
# model = get_network(patch_size=[168, 168, 16], spacing=[4.07, 4.07, 3.00])
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# batch_size = 1
# # device='meta' -> no memory is consumed for visualization
# model_graph = draw_graph(model, input_size=(batch_size, 168), device = device)
# model_graph.visual_graph

import torch
import torch.onnx
from model_maker import get_network

# Ensure the model is in evaluation mode
model = get_network(patch_size=[168, 168, 16], spacing=[4.07, 4.07, 3.00])
model.eval()

# Prepare a dummy input tensor with the right shape
# The shape must match the input shape the network expects
dummy_input = torch.randn(1, 1, 168, 168, 16)

# Define input and output names for clarity
input_names = ["input"]  
output_names = ["output"]  

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, "dynunet.onnx", input_names=input_names, output_names=output_names, opset_version=11)
