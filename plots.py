import tkinter as tk
import numpy as np

def plot_champ_temperature(T, cell_size=30, titre="Champ de température"):
    """
    Affiche un champ de température T (Nx, Ny) avec couleurs selon la valeur.
    
    Args:
        T : matrice 2D (Nx lignes, Ny colonnes)
        cell_size : taille d'une cellule en pixels
        titre : titre de la fenêtre
    """
    Nx, Ny = T.shape
    
    # Normaliser les valeurs de T pour les couleurs
    Tmin, Tmax = np.min(T), np.max(T)
    if Tmin == Tmax:
        Tmax = Tmin + 1e-5  # éviter division par zéro
    
    def valeur_to_couleur(val):
        """
        Convertit une valeur de T en couleur hexadécimale.
        Chaud = rouge, froid = bleu.
        """
        ratio = (val - Tmin) / (Tmax - Tmin)
        r = int(255 * ratio)
        g = 0
        b = int(255 * (1 - ratio))
        return f"#{r:02x}{g:02x}{b:02x}"
    
    # Création de la fenêtre Tkinter
    root = tk.Tk()
    root.title(titre)
    
    canvas = tk.Canvas(root, width=Ny*cell_size, height=Nx*cell_size)
    canvas.pack()
    
    # Dessiner chaque cellule
    for i in range(Nx):
        for j in range(Ny):
            couleur = valeur_to_couleur(T[i, j])
            x0 = j * cell_size
            y0 = i * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            canvas.create_rectangle(x0, y0, x1, y1, fill=couleur, outline="black")
    
    root.mainloop()

# --- Exemple d'utilisation ---
if __name__ == "__main__":
    Nx, Ny = 10, 15
    T = np.random.rand(Nx, Ny) * 100  # température aléatoire
    plot_champ_temperature(T, cell_size=20, titre="Test Champ de Température")
