from dotenv import load_dotenv
from mootlib import MootlibMatcher

# Load environment variables from .env file
load_dotenv()

def main():
    print("Hello from predict!")
    print("Initializing MootlibMatcher...")

    # Initialize the matcher
    matcher = MootlibMatcher()

    print("\nSearching for AI-related questions...")
    # Find similar questions about AI
    ai_questions = matcher.find_similar_questions(
        "Will AGI be achieved by 2025?",
        n_results=3,
        min_similarity=0.7
    )

    print("\nSearching for geopolitical questions...")
    # Find similar questions about geopolitics
    geo_questions = matcher.find_similar_questions(
        "Will China invade Taiwan in 2024?",
        n_results=3,
        min_similarity=0.7
    )

    print("\nSearching for Tesla stock questions...")
    # Find similar questions about Tesla
    tesla_questions = matcher.find_similar_questions(
        "Will Tesla stock reach $300 in 2024?",
        n_results=3,
        min_similarity=0.5
    )

    # Print results
    all_questions = ai_questions + geo_questions + tesla_questions

    print(f"\nFound {len(all_questions)} similar questions:")
    print("=" * 80)

    for i, q in enumerate(all_questions, 1):
        print(f"\n{i}. Question: {q.question}")
        print(f"   Platform: {q.source_platform}")
        print(f"   Similarity Score: {q.similarity_score:.3f}")
        print(f"   Current Probabilities: {q.formatted_outcomes}")
        if q.url:
            print(f"   Market URL: {q.url}")
        if q.n_forecasters:
            print(f"   Number of Forecasters: {q.n_forecasters}")
        if q.volume:
            print(f"   Volume: {q.volume}")
        print("-" * 40)

    # Access raw market data
    print(f"\nMarket Data Summary:")
    markets_df = matcher.markets_df
    print(f"Total markets: {len(markets_df)}")
    print("\nMarkets by platform:")
    platform_counts = markets_df["source_platform"].value_counts()
    for platform, count in platform_counts.items():
        print(f"  {platform}: {count}")

    # Get embeddings info
    embeddings_df = matcher.embeddings_df
    print(f"\nTotal questions with embeddings: {len(embeddings_df)}")


if __name__ == "__main__":
    print("hello?")
    main()
