import numpy as np
import pandas as pd
import os

def generate_dataset(n_sample=1000):
    np.random.seed(42)

    data=[]

    for i in range(n_sample):
        is_spam = np.random.choice([0, 1], p=[0.6, 0.4])


        if is_spam:
            has_link = np.random.choice([0, 1], p=[0.2, 0.8])
            has_money_words = np.random.choice([0, 1], p=[0.3, 0.7])
            has_urgent_words = np.random.choice([0, 1], p=[0.2, 0.8])
            has_caps = np.random.choice([0, 1], p=[0.3, 0.7])
            num_exclamation = np.random.randint(3, 15)
            email_length = np.random.randint(20, 120)
            num_digits = np.random.randint(2, 20)
            num_special_chars = np.random.randint(5, 30)

        else:
            
            has_link = np.random.choice([0, 1], p=[0.7, 0.3])
            has_money_words = np.random.choice([0, 1], p=[0.9, 0.1])
            has_urgent_words = np.random.choice([0, 1], p=[0.85, 0.15])
            has_caps = np.random.choice([0, 1], p=[0.8, 0.2])
            num_exclamation = np.random.randint(0, 4)
            email_length = np.random.randint(50, 500)
            num_digits = np.random.randint(0, 5)
            num_special_chars = np.random.randint(0, 10)


        data.append({
            'has_link':has_link,
            'has_money_words':has_money_words,
            'has_urgent_words':has_urgent_words,
            'has_caps':has_caps,
            'num_exclamation':num_exclamation,
            'email_length':email_length,
            'num_digits':num_digits,
            'num_special_chars':num_special_chars,
            'is_spam':is_spam
        })

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = generate_dataset(1000)
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), 'spam_email.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"Fake accounts: {df['is_spam'].sum()}")
    print(f"Real accounts: {len(df) - df['is_spam'].sum()}")

