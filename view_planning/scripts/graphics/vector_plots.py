import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

# Set up the vector field data
def create_vector_field():
    """Create an interesting vector field with varying magnitudes"""
    x = np.linspace(-3, 3, 20)
    y = np.linspace(-3, 3, 20)
    X, Y = np.meshgrid(x, y)
    
    # Create a spiral-like vector field
    U = -Y + 0.5 * X * np.exp(-0.3 * (X**2 + Y**2))
    V = X + 0.5 * Y * np.exp(-0.3 * (X**2 + Y**2))
    
    # Calculate magnitude for coloring
    M = np.sqrt(U**2 + V**2)
    
    return X, Y, U, V, M

# Define ALL the beautiful color schemes
def create_all_colorschemes():
    schemes = {}
    
    # === COSMIC & CELESTIAL ===
    schemes['Nebula'] = {
        'colors': ['#000000', '#4a148c', '#e91e63', '#ffffff'],
        'background': '#0a0a0a',
        'category': 'Cosmic'
    }
    
    schemes['Solar Flare'] = {
        'colors': ['#000000', '#cc4400', '#ffaa00', '#ffffff'],
        'background': '#000000',
        'category': 'Cosmic'
    }
    
    schemes['Galaxy'] = {
        'colors': ['#0d1421', '#3f51b5', '#9c27b0', '#c0c0c0'],
        'background': '#000000',
        'category': 'Cosmic'
    }
    
    schemes['Lunar Eclipse'] = {
        'colors': ['#2e2e2e', '#8b0000', '#cd853f', '#f0e68c'],
        'background': '#1a1a1a',
        'category': 'Cosmic'
    }
    
    # === ARTISTIC & PAINTERLY ===
    schemes['Van Gogh Starry Night'] = {
        'colors': ['#1a237e', '#3949ab', '#5c6bc0', '#ffeb3b'],
        'background': '#0d1421',
        'category': 'Artistic'
    }
    
    schemes['Monet Water Lilies'] = {
        'colors': ['#689f38', '#ba68c8', '#f8bbd9', '#f5f5dc'],
        'background': '#f9f9f9',
        'category': 'Artistic'
    }
    
    schemes['Rothko'] = {
        'colors': ['#8b0000', '#ff4500', '#ffd700'],
        'background': '#2f1b14',
        'category': 'Artistic'
    }
    
    schemes['Kandinsky'] = {
        'colors': ['#1976d2', '#d32f2f', '#ffeb3b'],
        'background': '#ffffff',
        'category': 'Artistic'
    }
    
    # === URBAN & INDUSTRIAL ===
    schemes['Cyberpunk'] = {
        'colors': ['#00ff41', '#00d4ff', '#ff00ff'],
        'background': '#000000',
        'category': 'Urban'
    }
    
    schemes['Concrete Jungle'] = {
        'colors': ['#616161', '#455a64', '#ffeb3b'],
        'background': '#212121',
        'category': 'Urban'
    }
    
    schemes['Traffic Flow'] = {
        'colors': ['#2e2e2e', '#ffa000', '#f44336', '#ffffff'],
        'background': '#1a1a1a',
        'category': 'Urban'
    }
    
    schemes['Blueprint'] = {
        'colors': ['#0d47a1', '#1976d2', '#42a5f5', '#bbdefb'],
        'background': '#ffffff',
        'category': 'Urban'
    }
    
    # === NATURAL & ORGANIC ===
    schemes['Bioluminescence'] = {
        'colors': ['#001122', '#006064', '#00bcd4', '#e0f7fa'],
        'background': '#000511',
        'category': 'Natural'
    }
    
    schemes['Autumn Leaves'] = {
        'colors': ['#2e7d32', '#ff8f00', '#ff5722', '#ffd54f'],
        'background': '#1b5e20',
        'category': 'Natural'
    }
    
    schemes['Coral Reef'] = {
        'colors': ['#0d47a1', '#26c6da', '#ff7043', '#ffffff'],
        'background': '#001f3f',
        'category': 'Natural'
    }
    
    schemes['Northern Lights'] = {
        'colors': ['#000000', '#00e676', '#2196f3', '#9c27b0'],
        'background': '#0a0a0a',
        'category': 'Natural'
    }
    
    # === LUXURY & PRECIOUS ===
    schemes['Gold Rush'] = {
        'colors': ['#000000', '#8d6e63', '#ffd700', '#fff8dc'],
        'background': '#0a0a0a',
        'category': 'Luxury'
    }
    
    schemes['Emerald'] = {
        'colors': ['#1b5e20', '#388e3c', '#00c853', '#c8e6c9'],
        'background': '#0d2818',
        'category': 'Luxury'
    }
    
    schemes['Sapphire'] = {
        'colors': ['#0d47a1', '#1976d2', '#2196f3', '#b3e5fc'],
        'background': '#001122',
        'category': 'Luxury'
    }
    
    schemes['Rose Gold'] = {
        'colors': ['#2e2e2e', '#e91e63', '#f8bbd9', '#fce4ec'],
        'background': '#1a1a1a',
        'category': 'Luxury'
    }
    schemes['Teal Gold'] = {
        'colors': ['#2e2e2e', '#008080', '#4db6ac', '#e0f2f1'],  # charcoal → teal → soft teal → pale mint
        'background': '#1a1a1a',
        'category': 'Luxury'
    }

    # === BOLD & EXPERIMENTAL ===
    schemes['Vaporwave'] = {
        'colors': ['#ff00ff', '#00ffff', '#9c27b0'],
        'background': '#1a0033',
        'category': 'Experimental'
    }
    
    schemes['Acid Trip'] = {
        'colors': ['#ff0080', '#00ff80', '#8000ff', '#ffff00'],
        'background': '#000000',
        'category': 'Experimental'
    }
    
    schemes['Miami Vice'] = {
        'colors': ['#ff1744', '#00e5ff', '#ffffff'],
        'background': '#0a0a0a',
        'category': 'Experimental'
    }
    
    schemes['Holographic'] = {
        'colors': ['#9c27b0', '#2196f3', '#00e676', '#ffeb3b'],
        'background': '#000000',
        'category': 'Experimental'
    }
    
    # === VINTAGE & RETRO ===
    schemes['Sepia Dreams'] = {
        'colors': ['#f5f5dc', '#daa520', '#8b4513', '#2f1b14'],
        'background': '#faf0e6',
        'category': 'Vintage'
    }
    
    schemes['Art Deco'] = {
        'colors': ['#ffd700', '#000000', '#f5deb3'],
        'background': '#1a1a1a',
        'category': 'Vintage'
    }
    
    schemes['70s Sunset'] = {
        'colors': ['#ff7043', '#e91e63', '#9c27b0'],
        'background': '#2e1065',
        'category': 'Vintage'
    }
    
    schemes['Film Noir'] = {
        'colors': ['#000000', '#424242', '#ffffff'],
        'background': '#0a0a0a',
        'category': 'Vintage'
    }
    
    # === SCIENTIFIC & TECHNICAL ===
    schemes['X-Ray'] = {
        'colors': ['#000000', '#00ff00', '#ffffff'],
        'background': '#000000',
        'category': 'Scientific'
    }
    
    schemes['Heat Map'] = {
        'colors': ['#0d47a1', '#2196f3', '#ffeb3b', '#f44336'],
        'background': '#000033',
        'category': 'Scientific'
    }
    
    schemes['Oscilloscope'] = {
        'colors': ['#000000', '#00ff00', '#80ff80'],
        'background': '#000000',
        'category': 'Scientific'
    }
    
    schemes['Spectral'] = {
        'colors': ['#9c27b0', '#2196f3', '#00e676', '#ffeb3b', '#ff5722'],
        'background': '#0a0a0a',
        'category': 'Scientific'
    }
    
    # === UNCONVENTIONAL ===
    schemes['Poison'] = {
        'colors': ['#000000', '#2e7d32', '#76ff03', '#ffffff'],
        'background': '#0a0a0a',
        'category': 'Unconventional'
    }
    
    schemes['Volcanic'] = {
        'colors': ['#212121', '#d32f2f', '#ff5722', '#ffeb3b'],
        'background': '#000000',
        'category': 'Unconventional'
    }
    
    schemes['Arctic'] = {
        'colors': ['#b3e5fc', '#ffffff', '#e1f5fe', '#f0f8ff'],
        'background': '#e3f2fd',
        'category': 'Unconventional'
    }
    
    schemes['Deep Forest'] = {
        'colors': ['#0d2818', '#2e7d32', '#76ff03', '#ffffff'],
        'background': '#051207',
        'category': 'Unconventional'
    }
    
    return schemes

def plot_vector_field_scheme(X, Y, U, V, M, scheme_name, colors, background, alpha=0.8):
    """Plot a single vector field with a specific color scheme"""
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list(scheme_name, colors, N=256)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)
    
    # Create the quiver plot with beautiful styling
    Q = ax.quiver(X, Y, U, V, M, 
                  cmap=cmap, 
                  scale=50, 
                  alpha=alpha,
                  width=0.003,
                  headwidth=3,
                  headlength=4,
                  angles='xy')
    
    # Style the plot
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    
    # Remove ticks and labels for clean look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set spine colors based on background
    dark_backgrounds = ['#000000', '#0a0a0a', '#0d1421', '#001122', '#212121', '#1a1a1a']
    spine_color = '#ffffff' if background in dark_backgrounds else '#333333'
    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(0.5)
    
    # Add title
    title_color = '#ffffff' if background in dark_backgrounds else '#333333'
    ax.set_title(scheme_name, fontsize=16, fontweight='bold', 
                color=title_color, pad=20, fontfamily='sans-serif')
    
    # Add colorbar with styling
    cbar = plt.colorbar(Q, ax=ax, shrink=0.8, aspect=30, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=title_color)
    cbar.outline.set_edgecolor(spine_color)
    cbar.ax.tick_params(labelcolor=title_color, labelsize=10)
    cbar.set_label('Vector Magnitude', color=title_color, fontsize=12)
    
    plt.tight_layout()
    return fig

def create_category_grid(schemes, category):
    """Create a comparison grid for a specific category"""
    X, Y, U, V, M = create_vector_field()
    
    # Filter schemes by category
    cat_schemes = {name: scheme for name, scheme in schemes.items() 
                   if scheme['category'] == category}
    
    if not cat_schemes:
        return None
    
    # Calculate grid dimensions
    n_schemes = len(cat_schemes)
    cols = min(4, n_schemes)
    rows = (n_schemes + cols - 1) // cols
    
    fig = plt.figure(figsize=(4 * cols, 3 * rows))
    fig.patch.set_facecolor('black')
    fig.suptitle(f'{category} Color Schemes', fontsize=20, color='white', y=0.98)
    
    for i, (name, scheme) in enumerate(cat_schemes.items()):
        ax = plt.subplot(rows, cols, i + 1)
        ax.set_facecolor(scheme['background'])
        
        # Create custom colormap
        cmap = LinearSegmentedColormap.from_list(name, scheme['colors'], N=256)
        
        # Plot quiver
        Q = ax.quiver(X, Y, U, V, M, 
                      cmap=cmap, 
                      scale=50, 
                      alpha=0.85,
                      width=0.004,
                      headwidth=3,
                      headlength=4)
        
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Style spines
        dark_backgrounds = ['#000000', '#0a0a0a', '#0d1421', '#001122', '#212121', '#1a1a1a']
        spine_color = '#ffffff' if scheme['background'] in dark_backgrounds else '#666666'
        for spine in ax.spines.values():
            spine.set_color(spine_color)
            spine.set_linewidth(1)
        
        # Title
        title_color = '#ffffff' if scheme['background'] in dark_backgrounds else '#333333'
        ax.set_title(name, fontsize=12, fontweight='bold', 
                    color=title_color, pad=8)
    
    plt.tight_layout()
    return fig

def showcase_best_schemes():
    """Showcase the most stunning schemes individually"""
    X, Y, U, V, M = create_vector_field()
    schemes = create_all_colorschemes()
    
    # Hand-picked best schemes for individual showcase
    showcase = ['Nebula', 'Solar Flare', 'Van Gogh Starry Night', 'Cyberpunk', 
               'Bioluminescence', 'Gold Rush', 'Vaporwave', 'Northern Lights']
    
    for name in showcase:
        if name in schemes:
            scheme = schemes[name]
            fig = plot_vector_field_scheme(X, Y, U, V, M, name, 
                                         scheme['colors'], scheme['background'])
            plt.show()
            print(f"✨ Showcased: {name}")

# Main execution
if __name__ == "__main__":
    print("🎨 ULTIMATE VECTOR FIELD COLOR COLLECTION")
    print("=" * 50)
    
    # Create vector field data
    X, Y, U, V, M = create_vector_field()
    schemes = create_all_colorschemes()
    
    print(f"Total schemes available: {len(schemes)}")
    
    # Show categories
    categories = list(set(scheme['category'] for scheme in schemes.values()))
    print(f"Categories: {', '.join(categories)}")
    
    # Ask user what to display
    print("\nChoose display option:")
    print("1. Showcase best schemes individually")
    print("2. Show all schemes by category")
    print("3. Show specific category")
    print("4. Show all schemes (warning: lots of plots!)")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == '1':
        showcase_best_schemes()
    
    elif choice == '2':
        for category in sorted(categories):
            print(f"\n📊 Creating {category} category grid...")
            fig = create_category_grid(schemes, category)
            if fig:
                plt.show()
                print(f"✓ Generated: {category} Grid")
    
    elif choice == '3':
        print(f"\nAvailable categories: {', '.join(sorted(categories))}")
        cat = input("Enter category name: ").strip()
        if cat in categories:
            fig = create_category_grid(schemes, cat)
            if fig:
                plt.show()
                print(f"✓ Generated: {cat} Grid")
        else:
            print("Category not found!")
    
    elif choice == '4':
        print("\n🚀 Generating ALL schemes (this will take a while)...")
        for name, scheme in schemes.items():
            fig = plot_vector_field_scheme(X, Y, U, V, M, name, 
                                         scheme['colors'], scheme['background'])
            plt.show()
            print(f"✓ Generated: {name}")
    
    print("\n🎨 Color collection complete!")
    
    # Display scheme details
    print("\n" + "="*60)
    print("COLOR SCHEME CATALOG:")
    print("="*60)
    for category in sorted(categories):
        print(f"\n🎯 {category.upper()} SCHEMES:")
        cat_schemes = {name: scheme for name, scheme in schemes.items() 
                      if scheme['category'] == category}
        for name, scheme in cat_schemes.items():
            print(f"  • {name}")
            print(f"    Colors: {scheme['colors']}")
            print(f"    Background: {scheme['background']}")