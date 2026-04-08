# Sign Language Interpreter

A Pydantic-based Sign Language Interpreter environment for the Meta OpenEnv hackathon.

## Project Structure

```
meta/
├── models/
│   └── sign_action.py      # Pydantic models for sign actions
├── utils/                  # Utility functions
├── tests/                  # Test files
├── src/                    # Main application code
├── env.py                  # Environment configuration and ASL signs
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Features

- **Pydantic-based SignAction Model**: Structured representation of sign language gestures
- **ASL Signs Dictionary**: Pre-configured dictionary with 33 ASL signs including visual twin pairs
- **Multimodal Disambiguation**: Audio hints and context queries for challenging sign pairs
- **Reward Scaling**: Difficulty-based rewards (Easy: 1.0, Medium: 1.5, Hard: 2.0)
- **Type Safety**: Full type annotations and validation
- **Extensible Design**: Easy to add new signs and features

Our environment implements Multimodal Disambiguation using both audio hints and context queries, pushing the boundaries of symbolic sign language interpretation.

## SignAction Model

The `SignAction` model includes:

- Basic sign information (ID, name, type)
- Hand configuration (shape, orientation, location)
- Movement details (type, direction, speed)
- Metadata (meaning, difficulty, description)
- Visual references (video/image URLs)

## Available ASL Signs

Currently includes 33 common ASL signs with advanced features:

**Basic Signs**: Hello, Thank You, Please, Yes, No, Help, Water, Eat, Drink, Sorry

**Extended Signs**: House, Car, Milk, Bread, Fish, Bird, Tree, Flower, Computer, Phone, Music, Dance, Sleep, Work

**Visual Twin Pairs** (with audio hints and context clues):
- **BAT** (animal) vs **BASEBALL** (sports equipment)
- **SQUASH_VEGETABLE** (vegetable) vs **SQUASH_SPORT** (sport)  
- **ORANGE_FRUIT** (fruit) vs **ORANGE_COLOR** (color)

## Usage

```python
from env import ENV_CONFIG, ASL_SIGNS
from models.sign_action import SignAction

# Get all signs
all_signs = ENV_CONFIG.get_all_signs()

# Get a specific sign
hello_sign = ENV_CONFIG.get_sign_by_id("hello")

# Search signs
greeting_signs = ENV_CONFIG.search_signs("greeting")

# Filter by difficulty
easy_signs = ENV_CONFIG.get_signs_by_difficulty(1)
```

## Installation

```bash
pip install -r requirements.txt
```

## Development

This project is designed for the Meta OpenEnv hackathon and provides a foundation for building sign language interpretation tools.
