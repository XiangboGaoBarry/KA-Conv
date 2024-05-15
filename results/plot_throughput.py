import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('results.csv')

# Clean the data
df['Model Size'] = df['Conv Layer'].apply(lambda x: 'Tiny' if 'tiny' in x else 'Small' if 'small' in x else 'Regular')
df['Throughput (image/s)'] = df['Throughput (image/s)'].astype(int)

# Define markers for different convolution layers
activation_funcs = df['Activation / Basis Functions'].unique()
markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'P', 'X', 'h']
activation_func_markers = {layer: marker for layer, marker in zip(activation_funcs, markers)}

# Define colors for different activation/activation_func functions
conv_layers = df['Conv Layer'].unique()
colors = sns.color_palette("bright", len(conv_layers))
conv_layer_colors = {func: color for func, color in zip(conv_layers, colors)}

# Plot settings
sns.set(style='whitegrid')
plt.figure(figsize=(16, 6))
# Plot each point with different marker and color
for (activation_func, conv_layer), group_data in df.groupby(['Activation / Basis Functions', 'Conv Layer']):
    plt.scatter(group_data['Accuracy (%)'], group_data['Throughput (image/s)'],
                label=f"{conv_layer}-{activation_func}",
                marker=activation_func_markers[activation_func], color=conv_layer_colors[conv_layer], s=300)

# Plot details
# plt.yscale('log')
# plt.xscale('log')
plt.ylabel('Throughput (image/s)')
plt.xlabel('Accuracy (%)')
plt.title('Model Performance Comparison')
plt.legend(title='Conv Layer and Activation/Basis Functions', 
           loc='upper right',
            bbox_to_anchor=(1.5, 1),
           )
plt.grid(True, which="both", ls="--")

# Top left indication
# plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.annotate('Better', xy=(0.2, 0.65), xycoords='axes fraction', fontsize=20)
plt.arrow(0.25, 0.58, -0.1, 0.1, transform=plt.gca().transAxes,
          length_includes_head=True, head_width=0.02, head_length=0.05, fc='black', ec='black')

plt.tight_layout()

plt.savefig('results.png')
plt.show()