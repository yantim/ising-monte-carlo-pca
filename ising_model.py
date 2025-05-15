import numpy as np
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt

class IsingModel:
    def __init__(self, L, lattice_type="square", T=1.0, steps=1000):
        self.L = L  # Size of the lattice (L x L)
        self.lattice_type = lattice_type  # Lattice type: "square" or "triangle"
        self.T = T  # Temperature
        self.Tc_square = 2.27  # Critical temperature for square lattice
        self.Tc_triangle = 3.65  # Critical temperature for triangular lattice
        self.steps = steps  # Number of steps (updates) to perform
        # Initialize the lattice with random spins (+1 or -1)
        self.lattice = np.random.choice([-1, 1], size=(L, L))
 
    def inverse_temperature(self):
        # Compute the inverse temperature (beta = 1 / T)
        return 1 / self.T
    
    def categorize_temperature(self):
        # Check if the temperature is low or high based on Tc
        if self.lattice_type == "square":
            Tc = self.Tc_square
        elif self.lattice_type == "triangle":
            Tc = self.Tc_triangle
        else:
            raise ValueError("Unsupported lattice type")
        
        if self.T < Tc:
            return "low"
        else:
            return "high"
        
    def random_site(self):
        # Select a random site on the lattice
        return np.random.randint(0, self.L), np.random.randint(0, self.L)

    def calculate_magnetization(self):
        """Calculate the magnetization of the system."""
        return np.sum(self.lattice)/ self.L**2

    def calculate_energy(self):
        """Calculate the energy of the system."""
        energy = 0
        for i in range(self.size):
            for j in range(self.size):
                neighbors = self.get_neighbors(i, j)
                energy += -self.lattice[i, j] * np.sum([self.lattice[n] for n in neighbors])
        return energy / 2  # To avoid double counting
    
    def get_neighbors(self, x, y):
        """
        Get the neighbors of a given spin (x, y) based on lattice type.
        Returns a list of neighbor coordinates.
        """
        neighbors = []
        
        if self.lattice_type == "square":
            neighbors = [
                ((x + 1) % self.L, y), ((x - 1) % self.L, y),  # right, left
                (x, (y + 1) % self.L), (x, (y - 1) % self.L)   # up, down
            ]
        elif self.lattice_type == "triangle":
            # Neighbors for a triangular lattice (6 neighbors)
            neighbors = [
                ((x + 1) % self.L, y), ((x - 1) % self.L, y),  # right, left
                (x, (y + 1) % self.L), (x, (y - 1) % self.L),  # up, down
                ((x + 1) % self.L, (y + 1) % self.L), ((x - 1) % self.L, (y - 1) % self.L)  # diagonal
            ]
        return neighbors                                                                           
        
    def metropolis_update(self):
        """
        Perform one Metropolis update step on the lattice.
        """
        x, y = np.random.randint(0, self.L, size=2)
        spin = self.lattice[x, y]
        
        # Get neighbors
        neighbors = self.get_neighbors(x, y)
        
        # Calculate the sum of neighbors
        neighbor_sum = sum(self.lattice[nx, ny] for nx, ny in neighbors)
        
        # Energy change if the spin flips
        delta_E = 2 * spin * neighbor_sum
        
        # Metropolis criterion: flip if energy decreases or with probability exp(-beta * delta_E)
        if delta_E <= 0 or np.random.rand() < np.exp(-self.inverse_temperature() * delta_E):
            self.lattice[x, y] *= -1
    
    def wolff_update(self):
        """
        Perform a single Wolff cluster update.
        """
        L = self.L
        x, y = np.random.randint(0, L, size=2)  # Randomly pick a seed spin
        spin = self.lattice[x, y]
        
        cluster = [(x, y)]
        visited = set(cluster)
        prob_add = 1 - np.exp(-2 * self.inverse_temperature())  # Probability to add a spin to the cluster
        
        while cluster:
            cx, cy = cluster.pop()
            neighbors = self.get_neighbors(cx, cy)
            for nx, ny in neighbors:
                if (nx, ny) not in visited and self.lattice[nx, ny] == spin:
                    if np.random.rand() < prob_add:
                        cluster.append((nx, ny))
                        visited.add((nx, ny))
        
        # Flip all spins in the cluster
        for vx, vy in visited:
            self.lattice[vx, vy] *= -1
    
    
    def simulate(self):
        # Generate sample for current temperature state
        temp_category = self.categorize_temperature()
     
        print(f"Temperature category: {temp_category} (T = {self.T})")

        # Perform the specified number of steps (updates)
        for step in range(self.steps):
            # print(f"Step {step + 1}/{self.steps}")
            
            if temp_category == "low":
                self.wolff_update()  # Perform Wolff update for low temperatures
            else:
                self.metropolis_update()  # Perform Metropolis update for high temperatures
                

    @staticmethod
    def generate_samples_parallel(L, temperatures, lattice_type, num_samples, save_dir, steps, num_workers=4):
        """
        Generate samples for a range of temperatures in parallel and save them as .npy files.

        Args:
            L (int): Lattice size.
            temperatures (list): List of temperatures.
            lattice_type (str): Type of lattice ('square' or 'triangular').
            num_samples (int): Number of samples to generate for each temperature.
            save_dir (str): Directory to save the generated samples.
            num_workers (int): Number of parallel workers.
        """
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

        # Use multiprocessing to parallelize temperature processing
        with Pool(processes=num_workers) as pool:
            # Use a partial function to pass fixed arguments to workers
            results = pool.starmap(
                generate_for_temperature,
                [(L, T, lattice_type, num_samples, save_dir,steps) for T in temperatures]
            )

    def plot_lattice(self):
        """Plot the final lattice configuration."""
        plt.figure(figsize=(6, 6))
        plt.imshow(self.lattice, cmap='bwr', interpolation='nearest')
        plt.colorbar(label="Spin")
        plt.title(f"Final Lattice Configuration (T = {self.T})")
        plt.show()
        
def generate_for_temperature(L, T, lattice_type, num_samples, save_dir,steps):
    """
    Generate samples for a single temperature.

    Args:
        L (int): Lattice size.
        T (float): Temperature.
        lattice_type (str): Type of lattice ('square' or 'triangular').
        num_samples (int): Number of samples to generate.
        save_dir (str): Directory to save the generated samples.
    """
    model = IsingModel(L=L, T=T, lattice_type=lattice_type,steps=steps)
    category = model.categorize_temperature()

    for sample_id in range(num_samples):
        # if category == "low":
        #     model.wolff_update()
        # else:
        #     model.metropolis_update()
        model.wolff_update()
        # Save the lattice configuration

        file_name = f"{T:.2f}_{sample_id}.npy"
        file_path = os.path.join(save_dir,file_name)
        np.save(file_path, model.lattice)

    

    
