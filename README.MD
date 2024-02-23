# GPT Data Generator

## Overview

The GPT Data Generator is a Python-based tool designed to generate JSON data for training Low-Rank Adaptation (LoRA) models with OpenAI's GPT-3. It utilizes OpenAI's API to fetch data based on user inputs, which can range from quotes from anime characters to dad jokes or country song lyrics. This project aims to simplify the process of data collection for machine learning projects, particularly those involving language models.

## Installation

### Prerequisites

- Python 3.6 or later
- An OpenAI API key

### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/prestonbourne/gpt_dataset_gen.git
    ```

2. Enter your OpenAI key
    - Create a `.env` file
    - Set your openai key

    ```plaintext
    OPENAI_API_KEY="<your-key-here>"
    ```

3. Install Dependencies
    - Ensure you're at the project root
    - Run

    ```bash
    pip install
    ```

4. Run the `main.py` file!
