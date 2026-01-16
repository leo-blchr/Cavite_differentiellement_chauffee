import tkinter as tk
import numpy as np

def plot_champ_temperature_brillant(T, cell_size=30, titre="Champ de température"):
    """
    Affiche un champ de température avec couleurs un peu brillantes :
    - Bleu foncé -> bleu clair pour les valeurs basses
    - Jaune -> rouge pour les valeurs hautes
    """
    Nx, Ny = T.shape

    Tmin, Tmax = np.min(T), np.max(T)
    if Tmin == Tmax:
        Tmax = Tmin + 1e-5

    def valeur_to_couleur(val):
        ratio = (val - Tmin) / (Tmax - Tmin)
        if ratio < 0.5:
            # Bleu foncé -> bleu clair (plus saturé)
            r = int(50 + ratio * 2 * 50)   # rouge léger
            g = int(50 + ratio * 2 * 50)   # vert léger
            b = int(180 + ratio * 2 * 75)  # bleu 180->255
        else:
            # Jaune -> rouge (plus saturé)
            ratio2 = (ratio - 0.5) * 2
            r = 255
            g = int(255 * (1 - ratio2))  # diminue vers rouge
            b = 0
        return f"#{r:02x}{g:02x}{b:02x}"

    # Création de la fenêtre
    root = tk.Tk()
    root.title(titre)
    canvas = tk.Canvas(root, width=Ny*cell_size+50, height=Nx*cell_size)
    canvas.pack()

    # Dessiner le champ
    for i in range(Nx):
        for j in range(Ny):
            couleur = valeur_to_couleur(T[i, j])
            x0, y0 = j * cell_size, i * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            canvas.create_rectangle(x0, y0, x1, y1, fill=couleur, outline="black")

    # Barre de couleur
    bar_x0, bar_x1 = Ny*cell_size + 10, Ny*cell_size + 30
    for k in range(100):
        ratio = k / 99
        val = Tmin + ratio * (Tmax - Tmin)
        couleur = valeur_to_couleur(val)
        y0 = Nx * (1 - ratio) * cell_size
        y1 = y0 - cell_size * Nx / 100
        canvas.create_rectangle(bar_x0, y0, bar_x1, y1, fill=couleur, outline="")

    canvas.create_text(bar_x1+20, Nx*cell_size-10, text=f"{Tmin:.1f}", anchor="w")
    canvas.create_text(bar_x1+20, 10, text=f"{Tmax:.1f}", anchor="w")

    root.mainloop()

# --- Exemple avec champ ordonné ---
if __name__ == "__main__":
    Nx, Ny = 10, 15
    T = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            T[i, j] = i + j  # gradient simple pour tester

    plot_champ_temperature_brillant(T, cell_size=25, titre="Champ Température Brillant")
