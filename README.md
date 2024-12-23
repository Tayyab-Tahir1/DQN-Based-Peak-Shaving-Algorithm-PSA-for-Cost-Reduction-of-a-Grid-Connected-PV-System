# DQN-Based Peak Shaving Algorithm (PSA) for Cost Reduction of a Grid-Connected PV System

## Overview

This repository presents a **Deep Q-Learning Network (DQN)** based **Peak Shaving Algorithm (PSA)** for optimizing the energy management of a bidirectional, grid-connected solar photovoltaic (PV) system with an integrated battery bank (PV+BB). Designed for residential settings in Oslo, Norway, the system leverages advanced machine learning to reduce energy costs, enhance efficiency, and enable sustainable energy trading.

The **Home Energy Management System (HEMS)** automates energy dispatch by accounting for varying load profiles, PV generation patterns, and utility structures like time-of-use tariffs and feed-in tariffs. The system’s peak shaving strategy effectively balances household energy demands while supporting demand charge management and energy trading.

---

## Features

- **Bidirectional Grid-Connected PV System**:
  - Optimizes energy dispatch between rooftop PV systems, battery banks, and the grid.
  - Supports home-to-grid (H2G) energy flow for enhanced energy trading.

- **Home Energy Management System (HEMS)**:
  - Automates energy dispatch evaluations using the **PSB-DQN algorithm**.
  - Prioritizes cost-effective energy sources to minimize household energy expenses.

- **Peak Shaving Strategy**:
  - Reduces demand charges by flattening energy usage during peak hours.
  - Enhances grid stability by mitigating voltage and frequency fluctuations.

- **DQN Algorithm**:
  - Outperforms traditional optimization methods in managing energy trading and demand charges.
  - Adapts to dynamic energy demands and environmental conditions.

---

## Methodology

### 1. System Design
- **Components**:
  - **Rooftop PV System**: Generates renewable energy tailored to Oslo’s climate.
  - **Battery Bank (BB)**: Stores energy for use during peak hours and low generation periods.
  - **Bidirectional Grid Connection**: Enables energy trading and supports grid stability.

- **Operational Modes**:
  - **Energy Storage**: Charges battery during off-peak hours or excess PV generation.
  - **Energy Dispatch**: Supplies stored energy during peak hours to reduce grid dependency.
  - **Energy Trading**: Sells excess PV generation back to the grid using feed-in tariffs.

### 2. PSB-DQN Algorithm
- Implements a Deep Q-Learning approach to automate energy dispatch decisions.
- Optimizes cost reduction by learning demand patterns and utility tariffs.
- Performs energy trading and peak shaving with minimal human intervention.

### 3. Evaluation Metrics
- **Financial Savings**:
  - Reduces energy costs by utilizing time-of-use tariffs and feed-in tariffs.
- **Energy Efficiency**:
  - Maximizes PV utilization and battery efficiency.
- **Environmental Impact**:
  - Reduces reliance on fossil fuels and associated greenhouse gas emissions.

---

## Results

### Key Findings
- **Cost Reduction**: DQN-based PSA significantly reduces household energy expenses compared to traditional methods.
- **Efficiency**: Demonstrates superior performance in peak shaving and energy trading scenarios.
- **Sustainability**: Enhances the adoption of renewable energy in residential settings, reducing carbon footprints.

### Visual Insights
- **Energy Flow Diagrams**: Illustrate PV, battery, and grid interactions.
- **Cost Analysis Charts**: Compare cost reductions achieved through DQN and baseline methods.
- **Battery Utilization Graphs**: Show state-of-charge trends over time.

---

## Prerequisites

- **Python (>= 3.8)**
- **TensorFlow (>= 2.6.0)**
- **NumPy**
- **pandas**
- **Matplotlib**
- **Seaborn**
- **TQDM**

---
