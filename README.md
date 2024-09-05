# Mutliple Trajectory Prediction


This repository contains the implementation of Multimodal Trajectory Prediction using deep convolutional networks for autonomous driving. The project focuses on predicting multiple possible future trajectories of traffic actors and estimating their probabilities. It is designed for self-driving vehicles (SDVs) and aims to improve their safety by accounting for the uncertainty and multimodality of traffic behavior.

## Overview

Autonomous vehicles must navigate dynamic environments with unpredictable behaviors from surrounding actors. This project presents a method for predicting multiple potential trajectories of surrounding vehicles, encoding the context of the environment using bird’s-eye view (BEV) raster images as input. This allows the system to predict future movements while considering factors such as roads, traffic rules, and interactions with other vehicles and pedestrians.

## Key Contributions:

Multimodal Prediction: Unlike single-trajectory predictions, this model infers multiple possible future paths and their associated probabilities.
Bird’s-Eye View Rasterization: The method transforms an actor’s surrounding environment into raster images to provide context for prediction.
Deep Convolutional Network (CNN) Architecture: A CNN model is used to predict multiple trajectories directly, allowing for more accurate long-term predictions.

## Abstract
Self-driving vehicles (SDVs) need to anticipate the possible movements of traffic actors around them to ensure safe and efficient driving. This project introduces a method that encodes an actor’s context into a raster image, which is then used by deep convolutional networks to predict future trajectories. By predicting multiple trajectories and their probabilities, this model helps address the uncertainty of dynamic environments.

## Features
Multimodal Prediction: Predict multiple future trajectories with associated probabilities.
BEV Rasterization: Raster images encode the actor’s surrounding context (map and other actors).
CNN-based Architecture: Efficient, deep convolutional model that directly computes multimodal predictions.

## Training

` python train.py --config config.yaml
`

