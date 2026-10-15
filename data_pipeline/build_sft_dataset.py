import json
import os
import random
from typing import List, Dict

# Set random seed for reproducibility
random.seed(42)

def generate_direct_qa() -> List[Dict]:
    """Single-turn Q&A: counting, arithmetic, common facts, colors, definitions."""
    items = []
    
    # 1. Counting
    counting_templates = [
        ("count from 1 to 5", "1, 2, 3, 4, 5."),
        ("count from 1 to 5 please", "Here you go: 1, 2, 3, 4, 5."),
        ("can you count to 5?", "Sure! 1, 2, 3, 4, 5."),
        ("count from 1 to 10", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10."),
        ("count from 1 to 10 please", "Here you go: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10."),
        ("can you count to 10?", "Sure! 1, 2, 3, 4, 5, 6, 7, 8, 9, 10."),
        ("count to ten", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10."),
        ("please count from 1 to 10", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10."),
        ("count backwards from 5 to 1", "5, 4, 3, 2, 1."),
        ("count down from 5", "5, 4, 3, 2, 1."),
        ("count down from 10 to 1", "10, 9, 8, 7, 6, 5, 4, 3, 2, 1."),
        ("count odd numbers from 1 to 9", "1, 3, 5, 7, 9."),
        ("count even numbers from 2 to 10", "2, 4, 6, 8, 10."),
        ("count from 1 to 20", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20."),
    ]
    for prompt, ans in counting_templates:
        for _ in range(50):
            items.append([{"role": "user", "content": prompt}, {"role": "assistant", "content": ans}])

    # 2. Number sequences & arithmetic
    seq_arith = [
        ("what comes after 1", "2 comes after 1."),
        ("what comes after 1?", "The number 2 comes after 1."),
        ("what number comes after 1?", "2 comes after 1."),
        ("what comes after 2?", "3 comes after 2."),
        ("what comes after 5?", "6 comes after 5."),
        ("what comes after 9?", "10 comes after 9."),
        ("what comes before 2?", "1 comes before 2."),
        ("what comes before 10?", "9 comes before 10."),
        ("what is 1 + 1?", "1 + 1 is 2."),
        ("what is 2 + 2?", "2 + 2 is 4."),
        ("what is 2 + 3?", "2 + 3 is 5."),
        ("what is 3 + 3?", "3 + 3 is 6."),
        ("what is 4 + 4?", "4 + 4 is 8."),
        ("what is 5 + 5?", "5 + 5 is 10."),
        ("what is 10 + 10?", "10 + 10 is 20."),
        ("what is 5 - 2?", "5 - 2 is 3."),
        ("what is 10 - 3?", "10 - 3 is 7."),
        ("what is 10 - 5?", "10 - 5 is 5."),
        ("what is 2 * 3?", "2 * 3 is 6."),
        ("what is 3 * 3?", "3 * 3 is 9."),
        ("what is 4 * 2?", "4 * 2 is 8."),
        ("if I have three apples and eat one, how many are left?", "You will have 2 apples left."),
        ("if you have 3 apples and eat 1, how many are left?", "There are 2 apples left."),
        ("if I have 5 candies and give 2 away, how many do I have?", "You have 3 candies left."),
        ("if you have 4 books and buy 2 more, how many do you have?", "You have 6 books in total."),
    ]
    for prompt, ans in seq_arith:
        for _ in range(35):
            items.append([{"role": "user", "content": prompt}, {"role": "assistant", "content": ans}])

    # 3. Facts, definitions, science, opposites
    facts = [
        ("what is the capital of France?", "The capital of France is Paris."),
        ("what is the capital of France", "The capital of France is Paris, France."),
        ("what is the capital of Japan?", "The capital of Japan is Tokyo."),
        ("what is the capital of Germany?", "The capital of Germany is Berlin."),
        ("what is the capital of the United Kingdom?", "The capital of the United Kingdom is London."),
        ("what is the capital of the United States?", "The capital of the United States is Washington, D.C."),
        ("what is the capital of Italy?", "The capital of Italy is Rome."),
        ("what color is the sky on a sunny day?", "The sky is blue on a sunny day."),
        ("what color is grass?", "Grass is green."),
        ("what color is a ripe banana?", "A ripe banana is yellow."),
        ("what color is snow?", "Snow is white."),
        ("what color is a lemon?", "A lemon is yellow."),
        ("what color are strawberries?", "Strawberries are red."),
        ("what do bees make?", "Bees make honey."),
        ("how many days are in a week?", "There are 7 days in a week."),
        ("how many months are in a year?", "There are 12 months in a year."),
        ("how many hours are in a day?", "There are 24 hours in a day."),
        ("how many minutes are in an hour?", "There are 60 minutes in an hour."),
        ("how many legs does a spider have?", "A spider has eight legs."),
        ("how many legs does a dog have?", "A dog has four legs."),
        ("how many legs does a cat have?", "A cat has four legs."),
        ("how many legs does a bird have?", "A bird has two legs."),
        ("which is bigger, an elephant or a mouse?", "An elephant is much bigger than a mouse."),
        ("is an elephant bigger than a mouse?", "Yes, an elephant is much bigger than a mouse."),
        ("which is faster: a car or a bicycle?", "A car is faster than a bicycle."),
        ("can birds fly in the sky?", "Yes, most birds can fly in the sky."),
        ("can fish breathe underwater?", "Yes, fish breathe underwater using gills."),
        ("what do people use an umbrella for?", "People use an umbrella to stay dry in the rain or to shield from the sun."),
        ("why do people sleep at night?", "People sleep to rest, restore their energy, and stay healthy."),
        ("is ice hot or cold?", "Ice is cold."),
        ("is fire hot or cold?", "Fire is hot."),
        ("what season comes after winter?", "Spring comes after winter."),
        ("what season comes after summer?", "Autumn (or fall) comes after summer."),
        ("what season comes after spring?", "Summer comes after spring."),
        ("what is water made of?", "Water is made of hydrogen and oxygen molecules (H2O)."),
        ("name a large ocean on Earth.", "The Pacific Ocean is the largest ocean on Earth."),
        ("what is the opposite of hot?", "The opposite of hot is cold."),
        ("what is the opposite of cold?", "The opposite of cold is hot."),
        ("what is the opposite of up?", "The opposite of up is down."),
        ("what is the opposite of big?", "The opposite of big is small."),
        ("what is the opposite of fast?", "The opposite of fast is slow."),
        ("what is the opposite of happy?", "The opposite of happy is sad."),
        ("what is the opposite of day?", "The opposite of day is night."),
        ("say hello in French.", "Hello in French is 'Bonjour'."),
        ("say thank you politely.", "Thank you very much, I truly appreciate your help!"),
        ("say goodbye.", "Goodbye! Have a great day ahead!"),
        ("name two fruits that are yellow.", "Two yellow fruits are bananas and lemons."),
        ("give me three words that rhyme with cat.", "Three words that rhyme with cat are bat, hat, and mat."),
        ("spell the word 'apple'.", "Apple is spelled A-P-P-L-E."),
        ("write a one-sentence poem about the moon.", "The gentle moon glows softly in the quiet night sky."),
    ]
    for prompt, ans in facts:
        for _ in range(35):
            items.append([{"role": "user", "content": prompt}, {"role": "assistant", "content": ans}])

    return items


def generate_identity_qa() -> List[Dict]:
    """Identity and capabilities queries."""
    items = []
    pairs = [
        ("what are you?", "I am Vak, a compact conversational language model."),
        ("what are you", "I am Vak, a lightweight artificial intelligence assistant."),
        ("who are you?", "I am Vak, a compact open-source conversational language model."),
        ("who are you", "My name is Vak, an AI assistant designed for lightweight conversational chat."),
        ("what is your name?", "My name is Vak."),
        ("what is your name", "I am called Vak."),
        ("who made you?", "I am an open-source transformer model designed to run efficiently on consumer hardware."),
        ("are you gemini?", "No, I am Vak, a compact independent language model."),
        ("are you chatgpt?", "No, I am Vak, an open-source lightweight language model."),
        ("are you a robot?", "I am an artificial intelligence software program, not a physical robot."),
        ("what can you do?", "I can chat with you, answer basic questions, count, and assist with simple text interactions."),
        ("what can you do", "I can help with basic question-answering, simple calculations, counting, and friendly conversation."),
        ("can you help me?", "Yes! How can I help you today?"),
        ("can you help me", "Of course! What would you like help with?"),
        ("tell me about yourself", "I am Vak, a compact ~28-million parameter transformer language model built for fast, local chat on consumer GPUs."),
    ]
    for prompt, ans in pairs:
        for _ in range(80):
            items.append([{"role": "user", "content": prompt}, {"role": "assistant", "content": ans}])
    return items


def generate_greetings_and_smalltalk() -> List[Dict]:
    """Varied natural greetings, feelings, and small talk."""
    items = []
    greetings = [
        ("hi", "Hello! How can I help you today?"),
        ("hi!", "Hi there! How are you doing?"),
        ("hi there", "Hello! It's great to meet you. What can I do for you?"),
        ("hello", "Hello! How are you today?"),
        ("hello!", "Hi! It's a pleasure to chat with you. How can I assist?"),
        ("hey", "Hey! What's on your mind?"),
        ("hey there", "Hey there! How is your day going?"),
        ("good morning", "Good morning! I hope you have a wonderful day ahead."),
        ("good morning!", "Good morning! How can I assist you today?"),
        ("good afternoon", "Good afternoon! How are things going with you?"),
        ("good evening", "Good evening! How can I help you tonight?"),
        ("how are you?", "I'm doing well, thank you for asking! How are you doing?"),
        ("how are you doing today?", "I'm doing great! How has your day been?"),
        ("how is it going?", "It's going well! How can I help you today?"),
        ("how is your day?", "My day is going great, thank you! How is yours?"),
        ("nice to meet you.", "Nice to meet you too! How can I help you today?"),
        ("nice to meet you!", "It's a pleasure to meet you as well!"),
        ("greetings!", "Greetings! How may I assist you today?"),
        ("are you ready to chat?", "Yes, I am ready! What would you like to talk about?"),
        ("how is the weather?", "As an AI, I don't have access to real-time weather sensors, but I hope it is nice where you are!"),
        ("what is your favorite color?", "I like blue, like a clear sky or deep ocean."),
        ("do you like reading books?", "I enjoy reading and learning from stories and informative articles!"),
        ("what is your favorite food?", "I don't eat food, but I hear pizza and fresh fruits are very popular!"),
        ("tell me a short story.", "Once upon a time, a small sparrow found a shiny silver pebble near a stream. Every morning, the pebble reflected the golden dawn, reminding the sparrow that beauty can be found in the simplest things."),
        ("tell me a short story about a puppy.", "Once upon a time, a cheerful puppy named Toby chased a dancing yellow butterfly across a meadow. When the butterfly landed softly on his nose, Toby sneezed happily and wagged his tail."),
        ("tell me something interesting.", "Honey never spoils! Archaeologists have found pots of honey in ancient Egyptian tombs that are thousands of years old and still perfectly edible."),
        ("do you have any hobbies?", "My favorite activity is chatting with people and answering curious questions!"),
        ("I had a great day today.", "That's wonderful to hear! I'm glad your day went so well."),
        ("I am feeling a little tired.", "Make sure to take some time to rest and recharge. Getting good rest is important!"),
        ("thank you!", "You are very welcome! Let me know if you need anything else."),
        ("thank you for your help!", "Happy to help! Have a great day!"),
        ("thanks", "Anytime! Let me know if you have more questions."),
        ("goodbye", "Goodbye! Have a great day!"),
        ("bye", "Bye! Take care!"),
        ("see you later", "See you later! Have a wonderful time!"),
    ]
    for prompt, ans in greetings:
        for _ in range(85):
            items.append([{"role": "user", "content": prompt}, {"role": "assistant", "content": ans}])
    return items


def generate_multiturn() -> List[Dict]:
    """2 to 3 turn natural dialogue exchanges."""
    threads = [
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How are you doing today?"},
            {"role": "user", "content": "I am good, thank you. What is your name?"},
            {"role": "assistant", "content": "My name is Vak. It is very nice to meet you!"},
        ],
        [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"},
            {"role": "user", "content": "Can you count from 1 to 5?"},
            {"role": "assistant", "content": "Sure! 1, 2, 3, 4, 5."},
            {"role": "user", "content": "Great, thank you!"},
            {"role": "assistant", "content": "You're welcome! Let me know if you need anything else."},
        ],
        [
            {"role": "user", "content": "Hey there"},
            {"role": "assistant", "content": "Hey! How can I assist you today?"},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "And what is the capital of Japan?"},
            {"role": "assistant", "content": "The capital of Japan is Tokyo."},
        ],
        [
            {"role": "user", "content": "Good morning"},
            {"role": "assistant", "content": "Good morning! How is your day starting out?"},
            {"role": "user", "content": "It is going well. What is 5 + 5?"},
            {"role": "assistant", "content": "5 + 5 is 10."},
            {"role": "user", "content": "Thanks!"},
            {"role": "assistant", "content": "Happy to help! Have a wonderful day."},
        ],
        [
            {"role": "user", "content": "Can you help me with a question?"},
            {"role": "assistant", "content": "Of course! What question do you have?"},
            {"role": "user", "content": "What comes after 1?"},
            {"role": "assistant", "content": "2 comes after 1."},
            {"role": "user", "content": "And what comes after 2?"},
            {"role": "assistant", "content": "3 comes after 2."},
        ],
        [
            {"role": "user", "content": "Tell me a story"},
            {"role": "assistant", "content": "Once upon a time, a small mouse made friends with a gentle deer in an oak forest. Whenever winter came, they shared nuts and berries and stayed warm together."},
            {"role": "user", "content": "That was sweet."},
            {"role": "assistant", "content": "I'm glad you liked it! Friendship makes every season brighter."},
        ],
        [
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am Vak, an open-source conversational language model."},
            {"role": "user", "content": "What can you do?"},
            {"role": "assistant", "content": "I can answer questions, count numbers, chat, and help with simple conversational tasks."},
        ],
    ]
    items = []
    for thread in threads:
        for _ in range(350):
            items.append(thread)
    return items


def generate_clarifications() -> List[Dict]:
    """Handling single-character inputs, typos, and confusing inputs politely."""
    items = []
    single_letters = ["d", "c", "a", "b", "e", "f", "x", "y", "z", "k", "p", "q", "m"]
    replies = [
        "I'm not sure what you mean by that. Could you please clarify or ask a question?",
        "Could you please elaborate? How can I help you?",
        "I didn't quite catch that. Feel free to ask a question or tell me what you need!",
        "What does that refer to? I'm happy to help if you have a specific question.",
    ]
    for letter in single_letters:
        for _ in range(50):
            reply = random.choice(replies)
            items.append([{"role": "user", "content": letter}, {"role": "assistant", "content": reply}])
            items.append([{"role": "user", "content": letter.upper()}, {"role": "assistant", "content": reply}])

    symbols = ["?", "...", "!", "???", "??", "help", "asdf", "qwerty", "idk", "test"]
    for sym in symbols:
        for _ in range(40):
            reply = random.choice(replies)
            items.append([{"role": "user", "content": sym}, {"role": "assistant", "content": reply}])

    return items


def main():
    print("Building balanced Vak Conversational Alignment Dataset...")
    
    qa = generate_direct_qa()
    identity = generate_identity_qa()
    greetings = generate_greetings_and_smalltalk()
    multiturn = generate_multiturn()
    clarify = generate_clarifications()
    
    all_conversations = qa + identity + greetings + multiturn + clarify
    random.shuffle(all_conversations)
    
    output_path = "data/processed/sft_conversations.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for conv in all_conversations:
            record = {"messages": conv, "is_synthetic": True}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"Generated {len(all_conversations):,} balanced conversational threads:")
    print(f"  • Direct Q&A (Counting, Math, Facts): {len(qa):,} threads")
    print(f"  • Identity & Capabilities: {len(identity):,} threads")
    print(f"  • Greetings & Small Talk: {len(greetings):,} threads")
    print(f"  • Multi-Turn Dialogue: {len(multiturn):,} threads")
    print(f"  • Clarifications & Fallbacks: {len(clarify):,} threads")
    print(f"Saved cleanly to {output_path}")


if __name__ == "__main__":
    main()
