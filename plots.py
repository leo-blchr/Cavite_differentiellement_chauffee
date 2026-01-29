import tkinter as tk
import numpy as np


def plot_champ_temperature_brillant(T, cell_size=30, titre="Champ de température"):
    import tkinter as tk
    import numpy as np

    Nx, Ny = T.shape
    Tmin, Tmax = np.min(T), np.max(T)
    if Tmin == Tmax:
        Tmax = Tmin + 1e-5

    def clamp(x):
        return max(0, min(255, int(x)))

    def valeur_to_couleur(val):
        ratio = (val - Tmin) / (Tmax - Tmin)
        ratio = max(0.0, min(1.0, ratio))

        # Gradient type viridis/plasma doux
        if ratio < 0.25:
            # bleu foncé → bleu clair
            t = ratio / 0.25
            r = 30 + 40 * t
            g = 50 + 80 * t
            b = 150 + 80 * t
        elif ratio < 0.5:
            # bleu → vert
            t = (ratio - 0.25) / 0.25
            r = 70 + 30 * t
            g = 130 + 80 * t
            b = 230 - 80 * t
        elif ratio < 0.75:
            # vert → jaune
            t = (ratio - 0.5) / 0.25
            r = 100 + 155 * t
            g = 210 + 45 * t
            b = 150 - 150 * t
        else:
            # jaune → rouge
            t = (ratio - 0.75) / 0.25
            r = 255
            g = 255 - 180 * t
            b = 0

        return f"#{clamp(r):02x}{clamp(g):02x}{clamp(b):02x}"

    # Fenêtre
    root = tk.Tk()
    root.title(titre)
    canvas = tk.Canvas(root, width=Ny*cell_size + 60, height=Nx*cell_size)
    canvas.pack()

    # Champ de température
    for i in range(Nx):
        for j in range(Ny):
            couleur = valeur_to_couleur(T[i, j])
            x0, y0 = j * cell_size, i * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            canvas.create_rectangle(x0, y0, x1, y1, fill=couleur, outline="")

    # Barre de couleur
    bar_x0, bar_x1 = Ny*cell_size + 15, Ny*cell_size + 35
    for k in range(100):
        ratio = k / 99
        val = Tmin + ratio * (Tmax - Tmin)
        couleur = valeur_to_couleur(val)
        y0 = Nx * (1 - ratio) * cell_size
        y1 = y0 - cell_size * Nx / 100
        canvas.create_rectangle(bar_x0, y0, bar_x1, y1, fill=couleur, outline="")

    canvas.create_text(bar_x1 + 10, Nx*cell_size - 5, text=f"{Tmin:.2f}", anchor="w")
    canvas.create_text(bar_x1 + 10, 5, text=f"{Tmax:.2f}", anchor="w")

    root.mainloop()


# --- Exemple avec champ ordonné ---
if __name__ == "__main__":
    Nx, Ny = 10, 15
    T = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            T[i, j] = i + j  # gradient simple pour tester

    plot_champ_temperature_brillant(T, cell_size=25, titre="Champ Température Brillant")
