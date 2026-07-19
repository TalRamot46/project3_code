import numpy as np
import matplotlib.pyplot as plt
import os

def greens_function(x, t, u0, D):
    """
    Green's function (fundamental solution) of the 1D diffusion equation:
    u(x,t) = u0 / sqrt(4 * pi * D * t) * exp(-x^2 / (4 * D * t))
    """
    return u0 / np.sqrt(4 * np.pi * D * t) * np.exp(-x**2 / (4 * D * t))

def main():
    # Parameters
    u0 = 1.0       # Source strength
    D = 0.5        # Diffusion coefficient
    times = [0.5, 1.5, 4.5]  # Three distinct time points
    
    # x grid for plotting (wide enough to capture diffusion at all times)
    x = np.linspace(-10, 10, 1000)
    
    # Modern, premium styling parameters
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'figure.titlesize': 22,
        'legend.fontsize': 12,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })
    
    # Create the figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Palette of colors for the three times
    colors = ["royalblue", "darkorange", "crimson"]  # Coral, Sky Blue, Forest Green
    # Line styles and widths to show overlap in the self-similar plot
    linestyles = ['-', '--', ':']
    linewidths = [3.5, 2.5, 1.5]
    
    # 1) Plot u(x, t) vs x
    for i, t in enumerate(times):
        u = greens_function(x, t, u0, D)
        ax1.plot(x, u, label=f't = $t_{i+1}$', color=colors[i], lw=2.0)
    
    ax1.set_xlabel('Lagrangian Position $m$', fontweight='bold')
    ax1.set_ylabel('Dimensional Value', fontweight='bold')
    ax1.set_title('Dimensional Profile $h(m,t)$', fontweight='bold', pad=15)
    ax1.grid(True)
    ax1.legend()
    
    # 2) Plot self-similar version
    # x-axis similarity variable: eta = x / sqrt(D * t)
    # y-axis similarity variable: u_scaled = u(x,t) / (u0 / sqrt(D * t))
    for i, t in enumerate(times):
        xi = x / np.sqrt(D * t)
        u = greens_function(x, t, u0, D)
        u_scaled = u / (u0 / np.sqrt(D * t))
        
        # We plot with different line widths/styles to make it clear they overlap perfectly
        ax2.plot(xi, u_scaled, label=f't = $t_{i+1}$', color=colors[i],
                 linestyle=linestyles[i], linewidth=linewidths[i])
        
    ax2.set_xlabel(r'Similarity Variable $\xi$', fontweight='bold')
    ax2.set_ylabel('Dimensionless Value', fontweight='bold')
    ax2.set_title(r'Self-similar profile $\tilde{h}(\xi)$', fontweight='bold', pad=15)
    ax2.set_xlim([-6, 6])  # Focus on the region where the Gaussian is non-zero
    ax2.grid(True)
    ax2.legend()
    
    # Add a main title explaining the physics
    # fig.suptitle('Illustration of Self-Similarity in 1D Diffusion', fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot
    output_filename = 'illustrate_self_similar.png'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_filename)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved successfully to: {output_path}")
    
    # Show the plot if running interactively
    # plt.show()

if __name__ == '__main__':
    main()
