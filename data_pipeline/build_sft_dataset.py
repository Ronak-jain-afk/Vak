import hashlib
import json
import os
import random
import re
from typing import Dict, List, Set, Tuple

# Set random seed for reproducibility
random.seed(42)

BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "held_out_benchmark.json")

def load_held_out_benchmark() -> List[Dict]:
    """Loads the canonical single source of truth for held-out evaluation prompts."""
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_text(t: str) -> str:
    """Normalize text for strict leakage checking.
    
    Strips peripheral punctuation for alphanumeric phrases, but preserves punctuation-only
    prompts (e.g. '...', '???', '!?') so distinct noise prompts are not collapsed to empty string.
    """
    t_clean = re.sub(r"\s+", " ", t).strip().lower()
    alphanumeric = re.sub(r"[^\w\s]", "", t_clean).strip()
    if alphanumeric:
        return re.sub(r"\s+", " ", alphanumeric).strip()
    return t_clean


def generate_arithmetic(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    """Generates varied arithmetic Q&A across operations, word problems, and division."""
    items = []
    
    # Addition up to 30
    for a in range(0, 31):
        for b in range(0, 31):
            if (a + b) % 2 == 0 or a < 12 or b < 5:
                res = a + b
                templates = [
                    (f"what is {a} + {b}?", f"{a} + {b} is {res}."),
                    (f"what is {a} plus {b}?", f"{a} plus {b} is {res}."),
                    (f"calculate {a} + {b}", f"{a} + {b} is {res}."),
                    (f"whats {a}+{b}", f"{a} + {b} is {res}."),
                ]
                for p, ans in templates:
                    if normalize_text(p) not in held_out_set:
                        items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])
    
    # Subtraction up to 35
    for a in range(1, 36):
        for b in range(0, a + 1):
            if (a * b) % 2 == 0 or b < 6 or (a - b) < 5:
                res = a - b
                templates = [
                    (f"what is {a} - {b}?", f"{a} - {b} is {res}."),
                    (f"what is {a} minus {b}?", f"{a} minus {b} is {res}."),
                    (f"calculate {a} - {b}", f"{a} - {b} is {res}."),
                ]
                for p, ans in templates:
                    if normalize_text(p) not in held_out_set:
                        items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    # Multiplication tables up to 15x15
    for a in range(1, 16):
        for b in range(1, 16):
            res = a * b
            templates = [
                (f"what is {a} * {b}?", f"{a} * {b} is {res}."),
                (f"what is {a} times {b}?", f"{a} times {b} is {res}."),
                (f"calculate {a} * {b}", f"{a} * {b} is {res}."),
            ]
            for p, ans in templates:
                if normalize_text(p) not in held_out_set:
                    items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    # Clean integer division
    div_pairs = [
        (4, 2), (6, 2), (6, 3), (8, 2), (8, 4), (9, 3), (10, 2), (10, 5),
        (12, 2), (12, 3), (12, 4), (12, 6), (14, 2), (14, 7), (15, 3), (15, 5),
        (16, 2), (16, 4), (16, 8), (18, 2), (18, 3), (18, 6), (18, 9), (20, 2),
        (20, 4), (20, 5), (20, 10), (21, 3), (21, 7), (24, 3), (24, 4), (24, 6),
        (24, 8), (24, 12), (25, 5), (27, 3), (27, 9), (28, 4), (28, 7), (30, 3),
        (30, 5), (30, 6), (30, 10), (32, 4), (32, 8), (35, 5), (35, 7), (36, 4),
        (36, 6), (36, 9), (36, 12), (40, 4), (40, 5), (40, 8), (40, 10), (42, 6),
        (42, 7), (45, 5), (45, 9), (48, 6), (48, 8), (48, 12), (49, 7), (50, 5),
        (50, 10), (54, 6), (54, 9), (56, 7), (56, 8), (60, 5), (60, 6), (60, 10),
        (60, 12), (63, 7), (63, 9), (64, 8), (70, 7), (70, 10), (72, 8), (72, 9),
        (80, 8), (80, 10), (81, 9), (90, 9), (90, 10), (100, 2), (100, 4), (100, 5),
        (100, 10), (100, 20), (100, 25), (100, 50), (120, 10), (120, 12), (144, 12)
    ]
    for num, den in div_pairs:
        res = num // den
        templates = [
            (f"what is {num} / {den}?", f"{num} divided by {den} is {res}."),
            (f"what is {num} divided by {den}?", f"{num} divided by {den} is {res}."),
            (f"calculate {num} / {den}", f"{num} / {den} = {res}."),
            (f"divide {num} by {den}", f"{num} divided by {den} equals {res}."),
        ]
        for p, ans in templates:
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    # Word problems
    names = ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry", "Ivy", "Jack", "Lucas", "Maya", "Noah", "Olivia", "Sophia"]
    items_list = ["apples", "books", "pencils", "stickers", "cookies", "marbles", "oranges", "candies", "pens", "erasers"]
    for i in range(len(names)):
        n1 = names[i]
        n2 = names[(i + 1) % len(names)]
        item = items_list[i % len(items_list)]
        for count1 in [5, 6, 8, 10, 12, 15, 20]:
            for count2 in [1, 2, 3, 4, 5]:
                if count1 > count2:
                    rem = count1 - count2
                    tot = count1 + count2
                    p1 = f"If {n1} has {count1} {item} and gives {count2} to {n2}, how many {item} does {n1} have left?"
                    a1 = f"{n1} has {rem} {item} left ({count1} - {count2} = {rem})."
                    if normalize_text(p1) not in held_out_set:
                        items.append([{"role": "user", "content": p1}, {"role": "assistant", "content": a1}])
                    
                    p2 = f"If {n1} has {count1} {item} and {n2} gives {n1} {count2} more, how many {item} does {n1} have in total?"
                    a2 = f"{n1} has {tot} {item} in total ({count1} + {count2} = {tot})."
                    if normalize_text(p2) not in held_out_set:
                        items.append([{"role": "user", "content": p2}, {"role": "assistant", "content": a2}])

                    p3 = f"Calculate the total if {n1} has {count1} {item} and receives {count2} more."
                    a3 = f"The total is {tot} {item} ({count1} + {count2} = {tot})."
                    if normalize_text(p3) not in held_out_set:
                        items.append([{"role": "user", "content": p3}, {"role": "assistant", "content": a3}])

                    p4 = f"Who has more {item} if {n1} has {count1} and {n2} has {count2}?"
                    a4 = f"{n1} has more {item} ({count1} is greater than {count2})."
                    if normalize_text(p4) not in held_out_set:
                        items.append([{"role": "user", "content": p4}, {"role": "assistant", "content": a4}])
    return items


def generate_counting(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    """Generates counting in various forward, backward, and skip-counting sequences."""
    items = []
    ranges = [
        (1, 5), (1, 10), (1, 15), (1, 20), (1, 25), (1, 30),
        (5, 10), (5, 15), (5, 20), (10, 15), (10, 20), (10, 25), (15, 20), (15, 25), (20, 25), (20, 30),
        (2, 6), (3, 8), (4, 9), (6, 11), (7, 12), (8, 14), (9, 15), (12, 18), (14, 20)
    ]
    for start, end in ranges:
        nums_str = ", ".join(str(n) for n in range(start, end + 1)) + "."
        templates = [
            (f"count from {start} to {end}", nums_str),
            (f"count from {start} to {end} please", f"Here you go: {nums_str}"),
            (f"can you count from {start} to {end}?", f"Sure! {nums_str}"),
            (f"please count from {start} to {end}", nums_str),
        ]
        for p, ans in templates:
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    # Backwards counting
    back_ranges = [(5, 1), (10, 1), (10, 5), (6, 2), (8, 3), (7, 1), (9, 4), (15, 10), (15, 5), (20, 15), (20, 10), (25, 20)]
    for start, end in back_ranges:
        nums_str = ", ".join(str(n) for n in range(start, end - 1, -1)) + "."
        templates = [
            (f"count backwards from {start} to {end}", nums_str),
            (f"count down from {start} to {end}", nums_str),
            (f"count backwards from {start} down to {end}", nums_str),
            (f"can you count backwards from {start} to {end}?", f"Here is the countdown: {nums_str}"),
        ]
        for p, ans in templates:
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    # Even / Odd sequences
    even_odd_ranges = [
        ("even", 2, 10, [2, 4, 6, 8, 10]),
        ("even", 2, 20, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]),
        ("even", 10, 20, [10, 12, 14, 16, 18, 20]),
        ("odd", 1, 9, [1, 3, 5, 7, 9]),
        ("odd", 1, 19, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]),
        ("odd", 11, 21, [11, 13, 15, 17, 19, 21]),
    ]
    for kind, s, e, seq in even_odd_ranges:
        s_str = ", ".join(str(n) for n in seq) + "."
        templates = [
            (f"count {kind} numbers from {s} to {e}", s_str),
            (f"what are the {kind} numbers between {s} and {e}?", f"The {kind} numbers are: {s_str}"),
            (f"list the {kind} numbers from {s} to {e}", s_str),
        ]
        for p, ans in templates:
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    # Skip counting by 5s and 10s
    skip_seqs = [
        ("count by 5s from 5 to 50", ", ".join(str(n) for n in range(5, 55, 5)) + "."),
        ("count by 10s from 10 to 100", ", ".join(str(n) for n in range(10, 110, 10)) + "."),
        ("count by 2s from 2 to 20", ", ".join(str(n) for n in range(2, 22, 2)) + "."),
    ]
    for p, ans in skip_seqs:
        for _ in range(3):
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    # Next / Previous
    for n in range(1, 35):
        next_n = n + 1
        prev_n = n - 1
        templates = [
            (f"what comes after {n}?", f"{next_n} comes after {n}."),
            (f"what comes after {n}", f"{next_n} comes after {n}."),
            (f"what number comes after {n}?", f"The number {next_n} comes after {n}."),
        ]
        if prev_n >= 0:
            templates.append((f"what comes before {n}?", f"{prev_n} comes before {n}."))
            templates.append((f"what comes before {n}", f"{prev_n} comes before {n}."))
        for p, ans in templates:
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    return items


def generate_facts(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    """Generates general facts, science, geography, and world knowledge Q&A."""
    items = []
    
    capitals = [
        ("France", "Paris", "Europe"), ("Japan", "Tokyo", "Asia"), ("Germany", "Berlin", "Europe"),
        ("the United Kingdom", "London", "Europe"), ("the United States", "Washington, D.C.", "North America"),
        ("Italy", "Rome", "Europe"), ("Spain", "Madrid", "Europe"), ("Canada", "Ottawa", "North America"),
        ("Australia", "Canberra", "Oceania"), ("India", "New Delhi", "Asia"), ("China", "Beijing", "Asia"),
        ("Brazil", "Brasília", "South America"), ("Egypt", "Cairo", "Africa"), ("Russia", "Moscow", "Europe/Asia"),
        ("South Korea", "Seoul", "Asia"), ("Mexico", "Mexico City", "North America"), ("Argentina", "Buenos Aires", "South America"),
        ("South Africa", "Pretoria", "Africa"), ("Greece", "Athens", "Europe"), ("Portugal", "Lisbon", "Europe"),
        ("Sweden", "Stockholm", "Europe"), ("Norway", "Oslo", "Europe"), ("Netherlands", "Amsterdam", "Europe"),
        ("Switzerland", "Bern", "Europe"), ("Belgium", "Brussels", "Europe"), ("Austria", "Vienna", "Europe"),
        ("Poland", "Warsaw", "Europe"), ("Turkey", "Ankara", "Asia/Europe"), ("Saudi Arabia", "Riyadh", "Asia"),
        ("Thailand", "Bangkok", "Asia"), ("Vietnam", "Hanoi", "Asia"), ("Indonesia", "Jakarta", "Asia"),
        ("Ireland", "Dublin", "Europe"), ("New Zealand", "Wellington", "Oceania"), ("Chile", "Santiago", "South America"),
        ("Colombia", "Bogotá", "South America"), ("Peru", "Lima", "South America"), ("Kenya", "Nairobi", "Africa"),
        ("Nigeria", "Abuja", "Africa"), ("Morocco", "Rabat", "Africa"), ("Finland", "Helsinki", "Europe"),
        ("Denmark", "Copenhagen", "Europe"), ("Singapore", "Singapore", "Asia"), ("Malaysia", "Kuala Lumpur", "Asia"),
        ("Philippines", "Manila", "Asia"), ("Czech Republic", "Prague", "Europe"), ("Hungary", "Budapest", "Europe"),
        ("Romania", "Bucharest", "Europe"), ("Ukraine", "Kyiv", "Europe"), ("Pakistan", "Islamabad", "Asia")
    ]
    for country, cap, cont in capitals:
        templates = [
            (f"what is the capital of {country}?", f"The capital of {country} is {cap}."),
            (f"what is the capital of {country}", f"The capital of {country} is {cap}."),
            (f"name the capital of {country}", f"{cap} is the capital of {country}."),
            (f"which city is the capital of {country}?", f"{cap} is the capital of {country}."),
            (f"what continent is {country} in?", f"{country} is in {cont}."),
            (f"on which continent is {country} located?", f"{country} is located in {cont}."),
        ]
        for p, ans in templates:
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    science_facts = [
        ("what do bees make?", "Bees make honey."),
        ("what is water made of?", "Water is made of hydrogen and oxygen (H2O)."),
        ("what planet is known as the Red Planet?", "Mars is known as the Red Planet."),
        ("which planet is the largest in our solar system?", "Jupiter is the largest planet in our solar system."),
        ("which planet is closest to the Sun?", "Mercury is closest to the Sun."),
        ("which planet is known for its prominent rings?", "Saturn is famous for its bright and prominent planetary rings."),
        ("what shines brightly in the sky during the day?", "The Sun shines brightly during the day."),
        ("what is the chemical formula for water?", "The chemical formula for water is H2O."),
        ("what is the chemical formula for carbon dioxide?", "The chemical formula for carbon dioxide is CO2."),
        ("what gas do humans breathe in to live?", "Humans breathe in oxygen."),
        ("what gas do plants absorb from the air?", "Plants absorb carbon dioxide from the air."),
        ("what process do plants use to make food from sunlight?", "Plants use photosynthesis to convert sunlight into chemical energy."),
        ("at what temperature does water freeze in Celsius?", "Water freezes at 0 degrees Celsius (32 degrees Fahrenheit)."),
        ("at what temperature does water boil in Celsius?", "Water boils at 100 degrees Celsius (212 degrees Fahrenheit)."),
        ("is ice hot or cold?", "Ice is cold."),
        ("is fire hot or cold?", "Fire is hot."),
        ("can fish breathe underwater?", "Yes, fish breathe underwater using gills."),
        ("can birds fly in the sky?", "Yes, most birds can fly in the sky."),
        ("which is bigger, an elephant or a mouse?", "An elephant is much bigger than a mouse."),
        ("which is faster: a car or a bicycle?", "A car is faster than a bicycle."),
        ("which is faster: light or sound?", "Light is vastly faster than sound. Light travels at roughly 300,000 km/s in a vacuum."),
        ("what force pulls objects toward the center of the Earth?", "Gravity is the force that pulls objects toward Earth's center."),
        ("how many days are in a week?", "There are 7 days in a week."),
        ("how many months are in a year?", "There are 12 months in a year."),
        ("how many hours are in a day?", "There are 24 hours in a day."),
        ("how many minutes are in an hour?", "There are 60 minutes in an hour."),
        ("how many seconds are in a minute?", "There are 60 seconds in a minute."),
        ("how many days are in a standard year?", "There are 365 days in a standard calendar year (366 in a leap year)."),
        ("how many legs does a spider have?", "A spider has eight legs."),
        ("how many legs does an insect have?", "An insect has six legs."),
        ("how many legs does a dog have?", "A dog has four legs."),
        ("how many legs does a bird have?", "A bird has two legs."),
        ("what season comes after winter?", "Spring comes after winter."),
        ("what season comes after summer?", "Autumn (or fall) comes after summer."),
        ("what season comes after spring?", "Summer comes after spring."),
        ("what season comes after autumn?", "Winter comes after autumn."),
        ("what color is grass?", "Grass is green."),
        ("what color is a ripe banana?", "A ripe banana is yellow."),
        ("what color is snow?", "Snow is white."),
        ("what color is a lemon?", "A lemon is yellow."),
        ("what color are strawberries?", "Strawberries are red."),
        ("what color is a carrot?", "Carrots are typically orange."),
        ("what color is the sky on a sunny day?", "The sky is blue on a sunny day."),
        ("what color is coal?", "Coal is black."),
        ("what do people use an umbrella for?", "People use an umbrella to stay dry in the rain or to shield from the sun."),
        ("why do people sleep at night?", "People sleep to rest, restore energy, and stay healthy."),
        ("why do people drink water?", "People drink water to stay hydrated and keep their bodies healthy."),
        ("what organ in the human body pumps blood?", "The heart pumps blood throughout the human body."),
        ("what organs do humans use to breathe?", "Humans use their lungs to breathe oxygen."),
        ("what is the largest ocean on Earth?", "The Pacific Ocean is the largest ocean on Earth."),
        ("how many continents are there on Earth?", "There are seven continents on Earth."),
        ("what is the tallest mountain in the world?", "Mount Everest is the tallest mountain in the world above sea level."),
        ("what is the largest desert on Earth?", "The Antarctic Desert is the largest desert on Earth (the Sahara is the largest hot desert)."),
        ("what is the longest river in the world?", "The Nile River (and Amazon River) is traditionally considered the longest river in the world."),
        ("what is the hardest natural substance on Earth?", "Diamond is the hardest known natural substance."),
        ("how many teeth does an adult human typically have?", "An adult human typically has 32 teeth (including wisdom teeth)."),
        ("what is the primary source of energy for Earth?", "The Sun is the primary source of energy for Earth."),
        ("what are the three main states of matter?", "The three main states of matter are solid, liquid, and gas."),
        ("what is the center of an atom called?", "The center of an atom is called the nucleus."),
        ("what particle orbits the nucleus of an atom?", "Electrons orbit the nucleus of an atom."),
        ("how many bones are in the adult human skeleton?", "An adult human skeleton consists of 206 bones."),
        ("what pigment gives plants their green color?", "Chlorophyll gives plants their green color and absorbs light for photosynthesis."),
        ("what instrument is used to measure temperature?", "A thermometer is used to measure temperature."),
        ("what instrument is used to tell time?", "A clock (or watch) is used to tell time."),
        ("what instrument is used to measure weight?", "A scale is used to measure weight."),
        ("what tool do people use to cut paper?", "Scissors are used to cut paper."),
        ("what do you wear on your hands in winter?", "Gloves or mittens are worn on hands to keep them warm."),
        ("what is the largest mammal on Earth?", "The blue whale is the largest mammal on Earth."),
        ("what is the fastest land animal?", "The cheetah is the fastest land animal."),
        ("what do herbivores eat?", "Herbivores eat plants and plant-based vegetation."),
        ("what do carnivores eat?", "Carnivores eat meat and other animals."),
        ("what do omnivores eat?", "Omnivores eat both plants and meat."),
        ("what is the boiling point of water in Fahrenheit?", "The boiling point of water is 212 degrees Fahrenheit."),
        ("what is the freezing point of water in Fahrenheit?", "The freezing point of water is 32 degrees Fahrenheit."),
    ]
    for p, ans in science_facts:
        variations = [
            (p, ans),
            (p.rstrip("?"), ans),
            (f"tell me, {p.lower()}", ans),
            (f"can you tell me {p.lower()}?", ans),
            (f"do you know {p.lower()}?", f"Yes! {ans}"),
        ]
        for var_p, var_ans in variations:
            if normalize_text(var_p) not in held_out_set:
                items.append([{"role": "user", "content": var_p}, {"role": "assistant", "content": var_ans}])

    return items


def generate_coding(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    """Generates Python coding, syntax, built-ins, and execution concepts."""
    items = []
    
    code_qa = [
        ("how do you print in Python?", "You use the `print()` function, for example: `print('Hello, world!')`."),
        ("write a hello world program in Python", "```python\nprint('Hello, world!')\n```"),
        ("how do you define a function in Python?", "You define a function using the `def` keyword:\n```python\ndef my_function():\n    return 'Hello'\n```"),
        ("write a Python function to add two numbers", "```python\ndef add(a, b):\n    return a + b\n```"),
        ("write a Python function to multiply two numbers", "```python\ndef multiply(a, b):\n    return a * b\n```"),
        ("write a Python function to subtract two numbers", "```python\ndef subtract(a, b):\n    return a - b\n```"),
        ("write a Python function to check if a number is even", "```python\ndef is_even(n):\n    return n % 2 == 0\n```"),
        ("write a Python function to check if a number is odd", "```python\ndef is_odd(n):\n    return n % 2 != 0\n```"),
        ("write a Python function to return the square of a number", "```python\ndef square(x):\n    return x ** 2\n```"),
        ("write a Python function to return the cube of a number", "```python\ndef cube(x):\n    return x ** 3\n```"),
        ("write a Python function to find the maximum of two numbers", "```python\ndef maximum(a, b):\n    return a if a > b else b\n```"),
        ("write a Python function to find the minimum of two numbers", "```python\ndef minimum(a, b):\n    return a if a < b else b\n```"),
        ("write a Python function to check if a string is empty", "```python\ndef is_empty(s):\n    return len(s) == 0\n```"),
        ("write a Python function to return the length of a list", "```python\ndef get_length(lst):\n    return len(lst)\n```"),
        ("what is a list in Python?", "A list in Python is an ordered, mutable collection of items enclosed in square brackets, e.g., `my_list = [1, 2, 3]`."),
        ("how do you create an empty list in Python?", "You can create an empty list using `[]` or `list()`."),
        ("how do you add an item to a list in Python?", "You use the `.append()` method, for example: `my_list.append(item)`."),
        ("how do you remove an item from a list in Python?", "You can use `.remove(item)` to remove by value, or `.pop(index)` to remove by index."),
        ("how do you find the length of a list in Python?", "You use the built-in `len()` function, for example: `len(my_list)`."),
        ("what is a dictionary in Python?", "A dictionary is a collection of key-value pairs enclosed in curly braces, e.g., `my_dict = {'key': 'value'}`."),
        ("how do you create an empty dictionary in Python?", "You can create an empty dictionary using `{}` or `dict()`."),
        ("how do you access a value in a dictionary in Python?", "You access a value using its key inside brackets, e.g., `my_dict['key']`, or using `.get('key')`."),
        ("how do you add a new key-value pair to a dictionary?", "Assign a value to the key: `my_dict['new_key'] = 'new_value'`."),
        ("what is a tuple in Python?", "A tuple is an ordered, immutable sequence of elements enclosed in parentheses, e.g., `my_tuple = (1, 2, 3)`."),
        ("what is the difference between a list and a tuple in Python?", "Lists are mutable (can be changed after creation), while tuples are immutable (cannot be modified after creation)."),
        ("how do you write a for loop in Python?", "```python\nfor i in range(5):\n    print(i)\n```"),
        ("how do you write a while loop in Python?", "```python\ncount = 0\nwhile count < 5:\n    print(count)\n    count += 1\n```"),
        ("how do you write an if-else statement in Python?", "```python\nif x > 0:\n    print('Positive')\nelse:\n    print('Non-positive')\n```"),
        ("what does len('hello') return in Python?", "It returns `5` because the string 'hello' has 5 characters."),
        ("what does 2 ** 3 mean in Python?", "`2 ** 3` computes 2 raised to the power of 3, which equals `8`."),
        ("what does 10 // 3 evaluate to in Python?", "`10 // 3` performs integer floor division and evaluates to `3`."),
        ("what does 10 % 3 evaluate to in Python?", "`10 % 3` computes the modulo (remainder) and evaluates to `1`."),
        ("how do you comment a single line in Python?", "You use the `#` symbol at the beginning of the comment."),
        ("how do you convert an integer to a string in Python?", "You use the `str()` function, for example: `str(42)` returns `'42'`."),
        ("how do you convert a string to an integer in Python?", "You use the `int()` function, for example: `int('42')` returns `42`."),
        ("what is a boolean in Python?", "A boolean is a data type with two possible truth values: `True` or `False`."),
        ("what does the range(5) function do in Python?", "`range(5)` generates a sequence of integers from 0 up to 4 (length 5)."),
        ("how do you check the type of a variable in Python?", "You use the built-in `type()` function, e.g., `type(x)`."),
        ("what is None in Python?", "`None` is a special constant in Python that represents the absence of a value or null state."),
        ("what does a ZeroDivisionError mean in Python?", "A `ZeroDivisionError` occurs when you attempt to divide a number by zero (e.g. `10 / 0`)."),
        ("what does a KeyError mean in Python?", "A `KeyError` occurs when you try to access a key that does not exist in a dictionary."),
        ("what does an IndexError mean in Python?", "An `IndexError` occurs when you try to access an index that is out of range for a sequence (like a list or tuple)."),
        ("what does a TypeError mean in Python?", "A `TypeError` occurs when an operation or function is applied to an object of an inappropriate type (such as adding a string and an integer)."),
        ("how do you read a text file in Python?", "You use `with open('filename.txt', 'r') as f:\n    text = f.read()`."),
        ("how do you write to a text file in Python?", "You use `with open('filename.txt', 'w') as f:\n    f.write('content')`."),
        ("what is a list comprehension in Python?", "A list comprehension is a concise syntax to construct lists, e.g., `[x * 2 for x in range(5)]`."),
        ("how do you import a module in Python?", "You use the `import` keyword, for example: `import math` or `from math import sqrt`."),
        ("what does the strip() method do in Python?", "The `.strip()` method removes whitespace characters from the beginning and end of a string."),
        ("what does the split() method do in Python?", "The `.split()` method splits a string into a list of substrings separated by a delimiter."),
        ("what does the join() method do in Python?", "The `.join()` method joins elements of an iterable with a separator string, e.g., `', '.join(['a', 'b'])`."),
        ("how do you reverse a list in Python?", "You can use `my_list.reverse()` in-place or `my_list[::-1]` to create a reversed copy."),
        ("what does bool(0) return in Python?", "`bool(0)` returns `False` because zero is falsy in Python."),
        ("what does bool(1) return in Python?", "`bool(1)` returns `True`."),
        ("what does bool('') return in Python?", "`bool('')` returns `False` because an empty string is falsy in Python."),
    ]
    for p, ans in code_qa:
        variations = [
            (p, ans),
            (f"can you tell me {p.lower()}", ans),
            (f"please explain {p.lower()}", ans),
            (f"in python, {p.lower()}", ans),
            (f"how to {p.lower().replace('how do you ', '').replace('how to ', '')}", ans),
        ]
        for var_p, var_ans in variations:
            if normalize_text(var_p) not in held_out_set:
                items.append([{"role": "user", "content": var_p}, {"role": "assistant", "content": var_ans}])
    
    # Python code output execution questions
    ops = [
        ("len([1, 2, 3, 4])", "4"),
        ("len(['cat', 'dog'])", "2"),
        ("len([])", "0"),
        ("len('')", "0"),
        ("len('python')", "6"),
        ("len('hello world')", "11"),
        ("sum([1, 2, 3])", "6"),
        ("sum([10, 20])", "30"),
        ("sum([5, 5, 5])", "15"),
        ("max(4, 7, 2)", "7"),
        ("min(4, 7, 2)", "2"),
        ("max([10, 50, 20])", "50"),
        ("min([10, 50, 20])", "10"),
        ("abs(-15)", "15"),
        ("abs(20)", "20"),
        ("type(5)", "<class 'int'>"),
        ("type('hello')", "<class 'str'>"),
        ("type([1, 2])", "<class 'list'>"),
        ("type({'a': 1})", "<class 'dict'>"),
        ("type((1, 2))", "<class 'tuple'>"),
        ("type(True)", "<class 'bool'>"),
        ("'a' in 'apple'", "True"),
        ("'z' in 'apple'", "False"),
        ("'cat' in ['cat', 'dog']", "True"),
        ("'bird' in ['cat', 'dog']", "False"),
        ("3 + 7", "10"),
        ("15 - 9", "6"),
        ("4 * 5", "20"),
        ("20 // 4", "5"),
        ("25 // 5", "5"),
        ("7 % 2", "1"),
        ("10 % 2", "0"),
        ("2 ** 4", "16"),
        ("3 ** 2", "9"),
        ("5 ** 2", "25"),
        ("'hello'.upper()", "'HELLO'"),
        ("'WORLD'.lower()", "'world'"),
        ("'apple'.startswith('a')", "True"),
        ("'apple'.endswith('e')", "True"),
        ("[1, 2] + [3, 4]", "[1, 2, 3, 4]"),
        ("'ha' * 3", "'hahaha'"),
    ]
    for expr, out in ops:
        templates = [
            (f"what is the output of `{expr}` in Python?", f"The output of `{expr}` is `{out}`."),
            (f"what does `{expr}` evaluate to in Python?", f"`{expr}` evaluates to `{out}`."),
            (f"in Python, what does `{expr}` return?", f"It returns `{out}`."),
            (f"calculate `{expr}` in python", f"`{expr}` evaluates to `{out}`."),
        ]
        for p, ans in templates:
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    return items


def generate_reasoning(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    """Generates comparisons, rhyming, text transformations, and instructions."""
    items = []
    
    rhymes = [
        ("cat", "bat, hat, mat, rat"),
        ("sun", "fun, run, bun, spun"),
        ("night", "bright, light, sight, fight"),
        ("tree", "free, bee, see, knee"),
        ("rain", "pain, gain, train, brain"),
        ("star", "far, car, bar, jar"),
        ("cold", "bold, gold, fold, told"),
        ("day", "say, play, way, stay"),
        ("blue", "true, clue, shoe, new"),
        ("ring", "sing, wing, king, spring"),
        ("lake", "make, bake, take, cake"),
        ("boat", "coat, float, goat, moat"),
        ("wall", "ball, call, fall, tall"),
        ("book", "look, took, cook, hook"),
        ("light", "bright, night, sight, white"),
    ]
    for word, rh_words in rhymes:
        templates = [
            (f"give me words that rhyme with '{word}'", f"Words that rhyme with '{word}' include: {rh_words}."),
            (f"what words rhyme with '{word}'?", f"Some words that rhyme with '{word}' are: {rh_words}."),
            (f"name three words that rhyme with {word}", f"Three words that rhyme with '{word}' are {rh_words.split(', ')[0]}, {rh_words.split(', ')[1]}, and {rh_words.split(', ')[2]}."),
            (f"can you find a rhyme for '{word}'?", f"Yes! '{rh_words.split(', ')[0]}' rhymes with '{word}'."),
        ]
        for p, ans in templates:
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    comparisons = [
        ("is a car faster than a person running?", "Yes, a car is generally much faster than a running person."),
        ("is an airplane faster than a bicycle?", "Yes, an airplane is far faster than a bicycle."),
        ("is an elephant heavier than a dog?", "Yes, an elephant is much heavier than a dog."),
        ("is the Sun hotter than ice?", "Yes, the Sun is extremely hot, whereas ice is cold."),
        ("is boiling water hotter than freezing water?", "Yes, boiling water (100°C) is much hotter than freezing water (0°C)."),
        ("is a mountain taller than a house?", "Yes, a mountain is much taller than a house."),
        ("is a whale bigger than a dolphin?", "Yes, most whales are significantly larger than dolphins."),
        ("is an ocean deeper than a swimming pool?", "Yes, an ocean is thousands of meters deep, far deeper than any swimming pool."),
        ("is a feather lighter than a rock?", "Yes, a feather is vastly lighter than a rock."),
        ("is winter colder than summer?", "Yes, winter is the coldest season, while summer is the warmest."),
        ("is an hour longer than a minute?", "Yes, an hour consists of 60 minutes, so it is much longer than a single minute."),
        ("is a year longer than a month?", "Yes, a year contains 12 months, making it much longer than one month."),
        ("is an adult usually taller than a toddler?", "Yes, an adult is significantly taller than a toddler."),
        ("is walking slower than driving?", "Yes, walking is slower than driving a car."),
        ("is a tiger bigger than a house cat?", "Yes, a tiger is far larger and heavier than a domestic cat."),
    ]
    for p, ans in comparisons:
        for _ in range(3):
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    text_transforms = [
        ("convert 'hello world' to uppercase", "HELLO WORLD"),
        ("convert 'PYTHON' to lowercase", "python"),
        ("convert 'GOOD MORNING' to lowercase", "good morning"),
        ("convert 'learning' to uppercase", "LEARNING"),
        ("convert 'artificial intelligence' to uppercase", "ARTIFICIAL INTELLIGENCE"),
        ("reverse the letters in the word 'cat'", "The reversed letters of 'cat' are 'tac'."),
        ("reverse the word 'dog'", "The reversed word is 'god'."),
        ("reverse the word 'star'", "The reversed word is 'rats'."),
        ("reverse the word 'stop'", "The reversed word is 'pots'."),
        ("reverse the word 'live'", "The reversed word is 'evil'."),
        ("how many letters are in the word 'apple'?", "The word 'apple' has 5 letters."),
        ("how many letters are in the word 'banana'?", "The word 'banana' has 6 letters."),
        ("how many letters are in the word 'sun'?", "The word 'sun' has 3 letters."),
        ("how many letters are in the word 'computer'?", "The word 'computer' has 8 letters."),
        ("how many letters are in the word 'book'?", "The word 'book' has 4 letters."),
        ("spell the word 'cat'", "C-A-T."),
        ("spell the word 'dog'", "D-O-G."),
        ("spell the word 'book'", "B-O-O-K."),
        ("spell the word 'tree'", "T-R-E-E."),
        ("spell the word 'smile'", "S-M-I-L-E."),
    ]
    for p, ans in text_transforms:
        templates = [
            (p, ans),
            (f"please {p.lower()}", ans),
            (f"can you {p.lower()}?", ans),
            (f"instruction: {p.lower()}", ans),
        ]
        for var_p, var_ans in templates:
            if normalize_text(var_p) not in held_out_set:
                items.append([{"role": "user", "content": var_p}, {"role": "assistant", "content": var_ans}])

    opposites_synonyms = [
        ("what is a synonym for 'happy'?", "A synonym for 'happy' is 'joyful' (or 'cheerful')."),
        ("what is a synonym for 'fast'?", "A synonym for 'fast' is 'quick' (or 'rapid')."),
        ("what is a synonym for 'smart'?", "A synonym for 'smart' is 'intelligent' (or 'clever')."),
        ("what is a synonym for 'big'?", "A synonym for 'big' is 'large' (or 'huge')."),
        ("what is a synonym for 'tiny'?", "A synonym for 'tiny' is 'small' (or 'miniature')."),
        ("what is an antonym of 'brave'?", "An antonym of 'brave' is 'cowardly' (or 'fearful')."),
        ("what is an antonym of 'generous'?", "An antonym of 'generous' is 'selfish' (or 'stingy')."),
        ("what is an antonym of 'ancient'?", "An antonym of 'ancient' is 'modern' (or 'new')."),
        ("what is an antonym of 'noisy'?", "An antonym of 'noisy' is 'quiet' (or 'silent')."),
        ("what is an antonym of 'rough'?", "An antonym of 'rough' is 'smooth'."),
        ("what is an antonym of 'heavy'?", "An antonym of 'heavy' is 'light'."),
        ("what is an antonym of 'early'?", "An antonym of 'early' is 'late'."),
        ("what is an antonym of 'clean'?", "An antonym of 'clean' is 'dirty'."),
        ("what is an antonym of 'bright'?", "An antonym of 'bright' is 'dim' (or 'dark')."),
        ("what is an antonym of 'simple'?", "An antonym of 'simple' is 'complex' (or 'complicated')."),
        ("what is an antonym of 'sharp'?", "An antonym of 'sharp' is 'dull' (or 'blunt')."),
        ("what is a synonym for 'wealthy'?", "A synonym for 'wealthy' is 'rich' (or 'affluent')."),
        ("what is a synonym for 'difficult'?", "A synonym for 'difficult' is 'hard' (or 'challenging')."),
        ("what is a synonym for 'calm'?", "A synonym for 'calm' is 'peaceful' (or 'tranquil')."),
        ("what is a synonym for 'tidy'?", "A synonym for 'tidy' is 'neat' (or 'clean')."),
        ("what is a synonym for 'begin'?", "A synonym for 'begin' is 'start'."),
        ("what is a synonym for 'finish'?", "A synonym for 'finish' is 'complete' (or 'end')."),
        ("if today is Monday, what day is tomorrow?", "If today is Monday, tomorrow is Tuesday."),
        ("if today is Friday, what day was yesterday?", "If today is Friday, yesterday was Thursday."),
        ("if today is Saturday, what day is tomorrow?", "If today is Saturday, tomorrow is Sunday."),
        ("what season is known for falling leaves?", "Autumn (or fall) is known for leaves changing colors and falling from deciduous trees."),
        ("what season is the hottest time of year?", "Summer is typically the hottest season of the year."),
    ]
    for p, ans in opposites_synonyms:
        templates = [
            (p, ans),
            (f"tell me {p.lower()}", ans),
            (f"can you tell me {p.lower()}?", ans),
            (f"please answer: {p.lower()}", ans),
        ]
        for var_p, var_ans in templates:
            if normalize_text(var_p) not in held_out_set:
                items.append([{"role": "user", "content": var_p}, {"role": "assistant", "content": var_ans}])

    poems = [
        ("write a short two-line poem about the moon", "The silver moon glows soft and bright,\nA gentle beacon in the night."),
        ("write a two-line poem about the sea", "The rolling waves crash on the shore,\nSinging of oceans evermore."),
        ("write a short poem about the sun", "Golden rays awake the day,\nChasing all the dark away."),
        ("write a two-line poem about rain", "Pitter-patter falls the rain,\nDancing on the windowpane."),
        ("write a two-line poem about stars", "Tiny jewels in velvety skies,\nTwinkling bright before our eyes."),
        ("write a short poem about flowers", "Petals unfold in morning light,\nPainting the garden warm and bright."),
        ("write a short poem about trees", "Tall green branches reach the sky,\nWhispering as the breezes sigh."),
    ]
    for p, ans in poems:
        for _ in range(4):
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    return items


def generate_continuations(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    """Generates multi-turn threads across tutoring, follow-up Q&A, and conversational sequences."""
    items = []

    resumptions = [
        (
            [{"role": "user", "content": "count from 1 to 10"},
             {"role": "assistant", "content": "1, 2, 3, 4, 5, 6, 7, 8, 9, 10."}],
            "continue to 20",
            "11, 12, 13, 14, 15, 16, 17, 18, 19, 20."
        ),
        (
            [{"role": "user", "content": "count from 1 to 10"},
             {"role": "assistant", "content": "1, 2, 3, 4, 5, 6, 7, 8, 9, 10."}],
            "keep going to 20",
            "11, 12, 13, 14, 15, 16, 17, 18, 19, 20."
        ),
        (
            [{"role": "user", "content": "count from 1 to 5"},
             {"role": "assistant", "content": "1, 2, 3, 4, 5."}],
            "continue to 10",
            "6, 7, 8, 9, 10."
        ),
        (
            [{"role": "user", "content": "count from 1 to 5"},
             {"role": "assistant", "content": "1, 2, 3, 4, 5."}],
            "keep going",
            "6, 7, 8, 9, 10."
        ),
        (
            [{"role": "user", "content": "count from 11 to 15"},
             {"role": "assistant", "content": "11, 12, 13, 14, 15."}],
            "continue to 20",
            "16, 17, 18, 19, 20."
        ),
        (
            [{"role": "user", "content": "count from 5 to 10"},
             {"role": "assistant", "content": "5, 6, 7, 8, 9, 10."}],
            "continue to 15",
            "11, 12, 13, 14, 15."
        ),
        (
            [{"role": "user", "content": "count from 1 to 5"},
             {"role": "assistant", "content": "1, 2, 3, 4, 5."}],
            "now backwards",
            "5, 4, 3, 2, 1."
        ),
    ]

    for prefix, user_next, asst_next in resumptions:
        for _ in range(6):
            thread = list(prefix) + [
                {"role": "user", "content": user_next},
                {"role": "assistant", "content": asst_next}
            ]
            items.append(thread)

    dialogues = [
        [
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "What is your name?"},
            {"role": "assistant", "content": "My name is Vak. I am a compact conversational model."},
            {"role": "user", "content": "Can you do math?"},
            {"role": "assistant", "content": "Yes! I can help with addition, subtraction, multiplication, and simple division."},
            {"role": "user", "content": "What is 7 times 8?"},
            {"role": "assistant", "content": "7 * 8 is 56."},
            {"role": "user", "content": "Thank you!"},
            {"role": "assistant", "content": "You're very welcome! Let me know if you need any other calculations."},
        ],
        [
            {"role": "user", "content": "Hello Vak"},
            {"role": "assistant", "content": "Hello! It is great to chat with you. How can I assist?"},
            {"role": "user", "content": "I am learning Python."},
            {"role": "assistant", "content": "That is wonderful! Python is a great programming language. What are you currently working on?"},
            {"role": "user", "content": "How do I print something?"},
            {"role": "assistant", "content": "You can use the `print()` function, like this: `print('Hello!')`."},
            {"role": "user", "content": "How do I make a list?"},
            {"role": "assistant", "content": "You make a list using square brackets, for example: `numbers = [1, 2, 3]`."},
            {"role": "user", "content": "Thanks!"},
            {"role": "assistant", "content": "You are very welcome! Happy coding!"},
        ],
        [
            {"role": "user", "content": "Can you count to 5?"},
            {"role": "assistant", "content": "Sure! 1, 2, 3, 4, 5."},
            {"role": "user", "content": "Can you keep going to 10?"},
            {"role": "assistant", "content": "6, 7, 8, 9, 10."},
            {"role": "user", "content": "Now count backwards from 5 to 1."},
            {"role": "assistant", "content": "5, 4, 3, 2, 1."},
            {"role": "user", "content": "Awesome, thanks."},
            {"role": "assistant", "content": "Glad I could help!"},
        ],
        [
            {"role": "user", "content": "What is the capital of Spain?"},
            {"role": "assistant", "content": "The capital of Spain is Madrid."},
            {"role": "user", "content": "And what is the capital of Portugal?"},
            {"role": "assistant", "content": "The capital of Portugal is Lisbon."},
            {"role": "user", "content": "And Italy?"},
            {"role": "assistant", "content": "The capital of Italy is Rome."},
            {"role": "user", "content": "What continent are they in?"},
            {"role": "assistant", "content": "All three countries—Spain, Portugal, and Italy—are located in Europe."},
        ],
        [
            {"role": "user", "content": "Tell me a short story about a puppy."},
            {"role": "assistant", "content": "Once upon a time, an energetic puppy found a red rubber ball in the green meadow. He bounded through the grass, wagging his tail happily all afternoon."},
            {"role": "user", "content": "What was the puppy's name?"},
            {"role": "assistant", "content": "His name was Barnaby, a golden pup with floppy ears and boundless curiosity."},
            {"role": "user", "content": "That was cute."},
            {"role": "assistant", "content": "I'm glad you liked it! Puppies bring a lot of joy."},
        ],
        [
            {"role": "user", "content": "Why is the sky blue?"},
            {"role": "assistant", "content": "The sky is blue because sunlight scatters off gases in Earth's atmosphere. Blue light scatters more than other colors because it travels as smaller, shorter waves."},
            {"role": "user", "content": "What about at sunset?"},
            {"role": "assistant", "content": "At sunset, sunlight travels through more atmosphere, scattering blue light away and letting red and orange light reach your eyes."},
            {"role": "user", "content": "That makes total sense."},
            {"role": "assistant", "content": "Nature has fascinating explanations! Feel free to ask more science questions anytime."},
        ],
        [
            {"role": "user", "content": "What are vowels in English?"},
            {"role": "assistant", "content": "The standard vowels in English are A, E, I, O, and U (and sometimes Y)."},
            {"role": "user", "content": "How many vowels are in the word 'education'?"},
            {"role": "assistant", "content": "The word 'education' contains all five standard vowels: e, u, a, i, and o."},
            {"role": "user", "content": "That's cool, it has all of them!"},
            {"role": "assistant", "content": "Yes! Words that contain all five vowels are called panvowel words."},
        ],
        [
            {"role": "user", "content": "I'm feeling bored today."},
            {"role": "assistant", "content": "I'm sorry to hear that! Would you like to hear a fun science fact, solve a riddle, or practice some trivia?"},
            {"role": "user", "content": "Tell me a fun science fact."},
            {"role": "assistant", "content": "Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible!"},
            {"role": "user", "content": "Wow, 3000 years old?"},
            {"role": "assistant", "content": "Yes! Honey's low moisture content and natural acidity prevent bacteria and microorganisms from growing."},
        ],
        [
            {"role": "user", "content": "Can you help me practice addition?"},
            {"role": "assistant", "content": "I'd love to! Here is one: what is 6 + 7?"},
            {"role": "user", "content": "It is 13."},
            {"role": "assistant", "content": "Correct! 6 + 7 is 13. Ready for another one?"},
            {"role": "user", "content": "Yes, give me another."},
            {"role": "assistant", "content": "What is 15 + 9?"},
            {"role": "user", "content": "24."},
            {"role": "assistant", "content": "Spot on! 15 + 9 is 24. Great job!"},
        ],
        [
            {"role": "user", "content": "What is the difference between a mammal and a reptile?"},
            {"role": "assistant", "content": "Mammals are warm-blooded, usually give live birth, have fur or hair, and nurse their young with milk. Reptiles are cold-blooded, have scaly skin, and typically lay eggs."},
            {"role": "user", "content": "Is a dolphin a mammal or a fish?"},
            {"role": "assistant", "content": "A dolphin is a mammal! Dolphins breathe air with lungs, give live birth, and nurse their calves with milk."},
            {"role": "user", "content": "What about a snake?"},
            {"role": "assistant", "content": "A snake is a reptile. Snakes have scales, are cold-blooded, and most lay eggs."},
        ]
    ]

    for d in dialogues:
        for _ in range(12):
            items.append(d)

    return items


def generate_identity(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    """Generates identity and capability queries."""
    items = []
    identity_templates = [
        ("what are you?", "I am Vak, a compact conversational language model."),
        ("what are you", "I am Vak, a compact conversational language model."),
        ("who are you?", "I am Vak, an open-source compact language model trained from scratch."),
        ("who are you", "I am Vak, an open-source compact language model."),
        ("what is your name?", "My name is Vak."),
        ("what is your name", "My name is Vak. I am a compact conversational model."),
        ("whats your name", "My name is Vak."),
        ("what's your name?", "My name is Vak."),
        ("who created you?", "I am Vak, designed and trained as an open-source compact language model."),
        ("who made you?", "I was trained as Vak, a parameter-efficient language model."),
        ("are you a human?", "No, I am Vak, an artificial intelligence language model."),
        ("what can you do?", "I can answer questions, count numbers, do simple math, write simple code, and chat."),
        ("what can you do", "I can answer questions, count numbers, do simple math, and engage in simple conversations."),
        ("what model are you?", "I am Vak, a compact decoder-only transformer language model."),
    ]
    for p, ans in identity_templates:
        for _ in range(6):
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])
    return items


def generate_greetings(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    """Generates varied greetings and conversational pleasantries."""
    items = []
    greetings = [
        ("hi", "Hello! How can I help you today?"),
        ("hi!", "Hi there! How can I help you today?"),
        ("hi there", "Hi there! How are you doing?"),
        ("hi assistant", "Hello! How can I help you today?"),
        ("hello", "Hello! How can I assist you today?"),
        ("hello!", "Hello! It is great to chat with you. How can I help?"),
        ("hello there", "Hello there! How can I help you?"),
        ("hello friend", "Hello! Wonderful to chat with you. How can I assist?"),
        ("hello assistant", "Hello! How can I assist you today?"),
        ("hello vak", "Hello! How can I help you today?"),
        ("hello to you", "Hello! Wonderful to meet you. What can I do for you?"),
        ("say hello", "Hello! It is great to chat with you."),
        ("hey", "Hey! What can I do for you today?"),
        ("hey there", "Hey there! How is your day going?"),
        ("hey friend", "Hey! How can I assist you today?"),
        ("good morning", "Good morning! How can I help you today?"),
        ("good morning!", "Good morning! I hope you have a great day ahead. How can I assist?"),
        ("good morning to you", "Good morning! I hope you have a wonderful day."),
        ("good evening", "Good evening! How can I help you tonight?"),
        ("good evening!", "Good evening! How can I assist you?"),
        ("how are you?", "I am doing well, thank you! How are you doing?"),
        ("how are you today?", "I'm doing great! How has your day been?"),
        ("how are you doing?", "I am doing well, thanks! How can I help?"),
        ("nice to meet you", "Nice to meet you too! How can I help you today?"),
        ("nice to meet you.", "Nice to meet you too! How can I help you today?"),
        ("thank you", "You're welcome! Let me know if you need anything else."),
        ("thanks!", "Happy to help! Let me know if you have more questions."),
        ("goodbye", "Goodbye! Have a wonderful day!"),
        ("bye!", "Bye! Take care!"),
        ("greetings", "Greetings! How may I help you today?"),
        ("greetings!", "Greetings! What can I assist you with today?"),
    ]
    for p, ans in greetings:
        for _ in range(5):
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])
    return items


def generate_fallbacks(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    """Generates polite clarification responses for typos, single characters, and noise."""
    items = []
    
    noise_inputs = [
        "d", "c", "x", "z", "q", "m", "p", "b", "j", "v",
        "D", "C", "X", "Z", "Q",
        "asdf", "qwerty", "zxcv", "idk", "blah",
        "???", "...", "!", "??", "!?",
        "test", "testing 123", "hello???",
    ]

    replies = [
        "I didn't quite catch that. Feel free to ask a question or tell me what you need!",
        "I'm not sure what you mean by that. Could you please clarify or ask a question?",
        "Could you please elaborate? How can I help you?",
        "What does that refer to? I'm happy to help if you have a specific question.",
    ]

    for noise in noise_inputs:
        for _ in range(5):
            reply = random.choice(replies)
            if normalize_text(noise) not in held_out_set:
                items.append([{"role": "user", "content": noise}, {"role": "assistant", "content": reply}])

    return items


def split_category_stratified(
    category_samples: List[List[Dict[str, str]]],
    val_ratio: float = 0.15
) -> Tuple[List[List[Dict[str, str]]], List[List[Dict[str, str]]]]:
    """
    Groups samples by normalized prompt cluster and splits clusters between Train and Val.
    Guarantees that:
    1) Every category is represented in both Train and Val.
    2) Train and Val have ZERO prompt overlap.
    """
    clusters: Dict[str, List[List[Dict[str, str]]]] = {}
    for sample in category_samples:
        prompt_norm = normalize_text(sample[0]["content"])
        if prompt_norm not in clusters:
            clusters[prompt_norm] = []
        clusters[prompt_norm].append(sample)

    cluster_keys = list(clusters.keys())
    random.seed(42)
    random.shuffle(cluster_keys)

    n_val_clusters = max(1, int(len(cluster_keys) * val_ratio))
    val_keys = set(cluster_keys[:n_val_clusters])
    train_keys = set(cluster_keys[n_val_clusters:])

    train_out = []
    val_out = []

    for k in train_keys:
        train_out.extend(clusters[k])
    for k in val_keys:
        val_out.extend(clusters[k])

    return train_out, val_out


def main():
    print("=" * 60)
    print("Building Stratified, Non-Overfitting, Leakage-Free Vak SFT Dataset")
    print("=" * 60)

    # 1. Load canonical held-out benchmark
    held_out_benchmark = load_held_out_benchmark()
    held_out_set = {normalize_text(item["prompt"]) for item in held_out_benchmark}
    print(f"Loaded {len(held_out_benchmark)} canonical held-out evaluation prompts from data/held_out_benchmark.json")

    # 2. Generate samples per category
    categories = {
        "arithmetic": generate_arithmetic(held_out_set),
        "counting": generate_counting(held_out_set),
        "continuations": generate_continuations(held_out_set),
        "greetings": generate_greetings(held_out_set),
        "identity": generate_identity(held_out_set),
        "facts": generate_facts(held_out_set),
        "coding": generate_coding(held_out_set),
        "reasoning": generate_reasoning(held_out_set),
        "fallbacks": generate_fallbacks(held_out_set),
    }

    train_all = []
    val_all = []

    print("\nStratified Category Partitioning (Cluster-based, 0 prompt leakage):")
    for cat_name, cat_samples in categories.items():
        tr, va = split_category_stratified(cat_samples, val_ratio=0.15)
        train_all.extend(tr)
        val_all.extend(va)
        print(f"  • {cat_name:15s}: {len(tr):5d} train, {len(va):4d} val (total {len(cat_samples):5d})")

    # Shuffle splits deterministically
    random.seed(42)
    random.shuffle(train_all)
    random.shuffle(val_all)

    # 3. Verify zero leakage and schema integrity
    train_prompts_norm = {normalize_text(s[0]["content"]) for s in train_all}
    val_prompts_norm = {normalize_text(s[0]["content"]) for s in val_all}

    inter = train_prompts_norm.intersection(val_prompts_norm)
    assert len(inter) == 0, f"Critical error: {len(inter)} prompts leaked between train and val: {inter}"

    inter_held_train = train_prompts_norm.intersection(held_out_set)
    assert len(inter_held_train) == 0, f"Critical error: {len(inter_held_train)} prompts leaked into held-out suite: {inter_held_train}"

    inter_held_val = val_prompts_norm.intersection(held_out_set)
    assert len(inter_held_val) == 0, f"Critical error: {len(inter_held_val)} val prompts leaked into held-out suite: {inter_held_val}"

    # Verify both splits contain core conversational vocabulary (e.g. hello, greetings, arithmetic, noise)
    hello_train = sum("hello" in s[0]["content"].lower() for s in train_all)
    hello_val = sum("hello" in s[0]["content"].lower() for s in val_all)
    
    noise_tokens = {"d", "c", "x", "z", "q", "m", "p", "b", "j", "v", "asdf", "qwerty", "zxcv", "idk", "blah", "???", "...", "!", "??", "!?"}
    noise_train = sum(normalize_text(s[0]["content"]) in noise_tokens or s[0]["content"] in noise_tokens for s in train_all)
    noise_val = sum(normalize_text(s[0]["content"]) in noise_tokens or s[0]["content"] in noise_tokens for s in val_all)

    assert hello_train > 0 and hello_val > 0, f"Stratification failed: hello_train={hello_train}, hello_val={hello_val}"
    assert noise_train > 0 and noise_val > 0, f"Stratification failed: noise_train={noise_train}, noise_val={noise_val}"

    print("\nSplit Verification:")
    print(f"  • Train total: {len(train_all):,} sequences ({len(train_prompts_norm):,} unique prompts)")
    print(f"  • Val total:   {len(val_all):,} sequences ({len(val_prompts_norm):,} unique prompts)")
    print(f"  • Total Sum:   {len(train_all) + len(val_all):,} sequences")
    print(f"  • Train/Val prompt overlap: 0 (Strictly Disjoint)")
    print(f"  • Overlap with Held-Out 50 Benchmark: 0 (Strictly Disjoint)")
    print(f"  • 'hello' distribution: {hello_train} in train, {hello_val} in val (Balanced)")
    print(f"  • noise distribution:   {noise_train} in train, {noise_val} in val (Balanced)")

    # 4. Save to files with is_synthetic: true provenance
    os.makedirs("data/processed", exist_ok=True)
    
    train_path = "data/processed/sft_train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_all:
            record = {"messages": s, "is_synthetic": True}
            f.write(json.dumps(record) + "\n")

    val_path = "data/processed/sft_val.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_all:
            record = {"messages": s, "is_synthetic": True}
            f.write(json.dumps(record) + "\n")

    # Backward compatibility
    conv_path = "data/processed/sft_conversations.jsonl"
    with open(conv_path, "w", encoding="utf-8") as f:
        for s in train_all:
            record = {"messages": s, "is_synthetic": True}
            f.write(json.dumps(record) + "\n")

    # 5. Compute SHA-256 and file sizes to update manifest
    def get_file_info(p: str):
        with open(p, "rb") as f:
            data = f.read()
        return hashlib.sha256(data).hexdigest(), len(data)

    train_sha, train_size = get_file_info(train_path)
    val_sha, val_size = get_file_info(val_path)
    conv_sha, conv_size = get_file_info(conv_path)
    bench_sha, bench_size = get_file_info("data/held_out_benchmark.json")

    manifest_path = "data/manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        manifest["artifacts"]["sft_train.jsonl"] = {
            "path": train_path,
            "description": "Supervised fine-tuning training sequences (stratified, is_synthetic=True)",
            "sha256": train_sha,
            "size_bytes": train_size,
            "sequence_count": len(train_all),
            "unique_prompts": len(train_prompts_norm)
        }
        manifest["artifacts"]["sft_val.jsonl"] = {
            "path": val_path,
            "description": "Supervised fine-tuning held-out validation split (disjoint from train)",
            "sha256": val_sha,
            "size_bytes": val_size,
            "sequence_count": len(val_all),
            "unique_prompts": len(val_prompts_norm)
        }
        manifest["artifacts"]["sft_conversations.jsonl"] = {
            "path": conv_path,
            "description": "Supervised fine-tuning training sequences (canonical training mirror)",
            "sha256": conv_sha,
            "size_bytes": conv_size,
            "sequence_count": len(train_all)
        }
        manifest["artifacts"]["held_out_benchmark.json"] = {
            "path": "data/held_out_benchmark.json",
            "description": "Canonical 50-prompt held-out generalization benchmark with grading criteria",
            "sha256": bench_sha,
            "size_bytes": bench_size,
            "prompt_count": len(held_out_benchmark)
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("Updated data/manifest.json with new SFT artifacts, exact checksums, and provenance.")

    print(f"\nFiles generated successfully:")
    print(f"  • {train_path} ({len(train_all):,} sequences, SHA: {train_sha[:8]}...)")
    print(f"  • {val_path} ({len(val_all):,} sequences, SHA: {val_sha[:8]}...)")


if __name__ == "__main__":
    main()
