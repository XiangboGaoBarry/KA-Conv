import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('results.csv')

# Clean the data
df['Model Size'] = df['Conv Layer'].apply(lambda x: 'Tiny' if 'tiny' in x else 'Small' if 'small' in x else 'Regular')
df['Parameters (B)'] = df['Parameters (B)'].str.replace(',', '').astype(int)

# Define markers for different convolution layers
conv_layers = df['Conv Layer'].unique()
markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'P', 'X', 'h']
conv_layer_markers = {layer: marker for layer, marker in zip(conv_layers, markers)}

# Define colors for different activation/basis functions
activation_funcs = df['Activation / Basis Functions'].unique()
colors = sns.color_palette("husl", len(activation_funcs))
activation_func_colors = {func: color for func, color in zip(activation_funcs, colors)}

# Plot settings
sns.set(style='whitegrid')
plt.figure(figsize=(16, 6))

# Plot each point with different marker and color
for (conv_layer, activation_func), group_data in df.groupby(['Conv Layer', 'Activation / Basis Functions']):
    plt.scatter(group_data['Accuracy (%)'], group_data['Parameters (B)'],
                label=f"{conv_layer} - {activation_func}",
                marker=conv_layer_markers[conv_layer], color=activation_func_colors[activation_func], s=300)

# Plot details
plt.yscale('log')
# plt.xscale('log')
plt.ylabel('Number of Parameters (B)')
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
plt.annotate('Better', xy=(0.15, 0.6), xycoords='axes fraction', fontsize=20)
plt.arrow(0.3, 0.58, -0.2, -0.1, transform=plt.gca().transAxes,
          length_includes_head=True, head_width=0.02, head_length=0.05, fc='black', ec='black')

plt.tight_layout()

plt.savefig('results.png')
plt.show()