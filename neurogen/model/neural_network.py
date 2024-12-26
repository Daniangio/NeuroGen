import random
import numpy as np
from neurogen.model.neuron import Neuron
from matplotlib import pyplot as plt

class NeuralNetwork:
    def __init__(self, num_neurons):
        self.neurons = [Neuron(str(idx)) for idx in range(num_neurons)]
        self.connect_neurons()

    def connect_neurons(self):
        for idx, neuron in enumerate(self.neurons):
            for _ in range(random.randint(1, 2)):
                try:
                    target_neuron = random.choice(self.neurons[idx+1:])
                    if target_neuron == neuron:
                        continue
                    if target_neuron.num_dendrites == 0 or random.random() < 0.5:
                        dendrite_index = None
                    else:
                        dendrite_index = random.randint(0, target_neuron.num_dendrites - 1)
                    neuron.connect(target_neuron, dendrite_index=dendrite_index)
                except:
                    pass

    @property
    def all_unique_neurons_and_dendrites(self):
        unique_neurons = set(self.neurons)
        unique_dendrites = set(dendrite for neuron in self.neurons for dendrite in neuron.dendrites)
        return list(unique_neurons.union(unique_dendrites))

    def step(self, dt):
        for neuron in self.all_unique_neurons_and_dendrites:
            neuron.integrate(dt)
        for neuron in self.all_unique_neurons_and_dendrites:
            neuron.update()
    
    def plot(self):
        fig, ax = plt.subplots()
        neuron_positions = {}

        # Plot neurons
        num_neurons = len(self.neurons)
        radius = 5
        for i, neuron in enumerate(self.neurons):
            angle = 2 * np.pi * i / num_neurons
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            neuron_positions[neuron.idx] = (x, y)
            ax.add_patch(plt.Circle((x, y), 0.5, color='blue', fill=False))
            plt.text(x, y, neuron.idx, fontsize=12, ha='center')

            # Plot dendrites in a circular arrangement around the neuron
            num_dendrites = len(neuron.dendrites)
            for j, dendrite in enumerate(neuron.dendrites):
                angle = 2 * np.pi * j / num_dendrites
                dx = x + 0.2 * np.cos(angle)
                dy = y + 0.2 * np.sin(angle)
                neuron_positions[dendrite.idx] = (dx, dy)
                ax.add_patch(plt.Circle((dx, dy), 0.2, color='green', fill=False))

        # Plot connections with arrows        
        for neuron in self.all_unique_neurons_and_dendrites:
            if hasattr(neuron, 'connected_to'):
                for target_neuron in neuron.connected_to:
                    if target_neuron:
                        x1, y1 = neuron_positions[neuron.idx]
                        x2, y2 = neuron_positions[target_neuron.idx]
                        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                                        arrowprops=dict(arrowstyle="->", color='k'))

        ax.set_aspect('equal', 'box')
        plt.xlim(-radius - 1, radius + 1)
        plt.ylim(-radius - 1, radius + 1)
        plt.show()