import pygame
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, Conv3D, Flatten, Dense, BatchNormalization, LSTM, TimeDistributed, Activation
from tensorflow.keras.models import clone_model

import pygad.kerasga

from games.christmas_jump.v3_inc import run_game

# Global var of the model
model = None

# Testing
def tiny_cnn(input_shape):

    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(4, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(BatchNormalization())

    # Flatten the output
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(16, activation='relu'))

    # Output layer with 2 neurons (for "jump" and "fall") and sigmoid activation
    model.add(Dense(3, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', # Changed to binary_crossentropy
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

# Time-dist net (too big)
def timedist_net():

    model = Sequential()

    # TimeDistributed CNN layers with Batch Normalization
    model.add(TimeDistributed(Conv2D(8, (3, 3), activation='relu'), input_shape=(4, 75, 75, 1)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Flatten()))

    # LSTM layer
    model.add(LSTM(32, activation='tanh'))

    # Fully connected layers
    model.add(Dense(16, activation='relu'))

    # Output layer
    model.add(Dense(3, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

# CONVlstm net (doesn't converge?) RUNNING
def convlstm_net():

    model = Sequential()

    # ConvLSTM2D layers with Batch Normalization
    model.add(ConvLSTM2D(16, (3, 3), activation='relu', input_shape=(4, 75, 75, 1), return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())

    # Flatten the output
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(32, activation='relu'))

    # Output layer
    model.add(Dense(3, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

# CONV3d net (doesn't converge)
def conv3d_net():

    model = Sequential()

    # Conv3D layers with Batch Normalization
    model.add(Conv3D(8, (2, 3, 3), activation='relu', input_shape=(4, 75, 75, 1)))
    model.add(BatchNormalization())
    model.add(Conv3D(16, (2, 3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv3D(32, (2, 3, 3), activation='relu'))
    model.add(BatchNormalization())

    # Flatten the output
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(32, activation='relu'))

    # Output layer
    model.add(Dense(3, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

# Works
def stable_net(input_shape):
    
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # Flatten the output
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(32, activation='relu'))

    # Output layer with 2 neurons (for "jump" and "fall") and sigmoid activation
    model.add(Dense(3, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', # Changed to binary_crossentropy
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

### TRAINING FUNCTIONS ### 
def fitness_func(ga_instance, solutions, solutions_indices):

    global model

    model_list = []
    batch_fitness = []

    for solution in solutions:

        new_model = clone_model(model)

        # Set the weights with the current solution
        model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=new_model,
                                                                weights_vector=solution)

        new_model.set_weights(weights=model_weights_matrix)

        model_list.append(new_model)

    results = run_game(0, model_list)

    for score, n_jumps in results:
        fitness = 0
        if n_jumps != 0:
            fitness = score * score / n_jumps
        
        batch_fitness.append(fitness)

    print("Batch Fitness: " + str(batch_fitness))
    return batch_fitness

def on_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(ga_instance.last_generation_fitness)[1]), end='\n\n')

## MAIN
def main(pretrained_model_path=None):

    global model

    # Load an existing model or create a new one
    if pretrained_model_path:
        model = keras.models.load_model(pretrained_model_path)
    else:
        model = tiny_cnn((50, 50, 1))

    model.summary()

    try:

        # Create the KERAS GA object
        keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=7)

        # Check if a pre-trained model is used
        if pretrained_model_path:

            # Get the current model weights
            current_weights = model.get_weights()

            # Convert the weights to a single flattened vector
            current_weights_vector = np.concatenate([weight.flatten() for weight in current_weights])

            # Replicate the weights vector for the entire initial population
            initial_population = np.array([current_weights_vector for _ in range(keras_ga.num_solutions)])
        else:
            # Use default initialization
            initial_population = keras_ga.population_weights

        # Trainer object
        ga_instance = pygad.GA(num_generations=250,
                            num_parents_mating=2,
                            fitness_func=fitness_func,
                            fitness_batch_size=6,
                            initial_population=initial_population,
                            mutation_probability=0.10,
                            on_generation=on_generation,
                            suppress_warnings=True)

        # Running the genetic algorithm
        ga_instance.run()

        # Plotting the training process
        fig = ga_instance.plot_fitness(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)
        fig.savefig("GA_Training_Process.png")  # Save the plot to a file

    except KeyboardInterrupt:

        print("\nCtrl+C pressed. Saving the current best model...")

        # Plotting the training process
        fig = ga_instance.plot_fitness(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)
        fig.savefig("GA_Training_Process.png")  # Save the plot to a file

        # Retrieve and save the current best solution
        solution, solution_fitness, _ = ga_instance.best_solution()
        best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
        model.set_weights(weights=best_solution_weights)
        model.save("best_model_interrupted.keras")

        print("Best model saved as 'best_model_interrupted.keras'. Exiting.")
        sys.exit(0)

    # Retrieving and saving the final best solution
    solution, solution_fitness, _ = ga_instance.best_solution()
    best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
    model.set_weights(weights=best_solution_weights)
    model.save("best_model.keras")

    # Print information about the best solution
    print("Parameters of the best solution:\n{solution}".format(solution=solution), end="\n\n")
    print("Length of the solution is:", len(solution), end='\n\n')
    print("Fitness value of the best solution:\n{solution_fitness}".format(solution_fitness=solution_fitness), end='\n\n')

if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser(description='Run the training process with an optional pre-trained model.')

    # Add argument for the pre-trained model path
    parser.add_argument('--model_path', type=str, default=None, help='Path to a pre-trained model. If not provided, a new model will be created.')

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function with the provided model path
    main(pretrained_model_path=args.model_path)