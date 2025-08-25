import os
import pandas as pd
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey, Text, Enum
import json

# --- Constants ---
ENRICHED_CSV = "data/mcq_dataset_enriched.csv"
# Get database connection details from environment variables
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "postgres")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def main():
    # Create a SQLAlchemy engine
    try:
        engine = create_engine(DATABASE_URL)
        metadata = MetaData()
    except Exception as e:
        print(f"Error creating database engine: {e}")
        return

    # Define the table structures
    categories = Table('categories', metadata,
        Column('id', String, primary_key=True),
        Column('name', String, unique=True, nullable=False)
    )

    questions = Table('questions', metadata,
        Column('id', String, primary_key=True),
        Column('title', String),
        Column('body', Text),
        Column('difficulty', Enum('EASY', 'MEDIUM', 'HARD', name='difficulty_enum'), nullable=False),
        Column('category_id', String, ForeignKey('categories.id')),
        Column('tags_text', Text) # Using Text to store comma separated tags
    )

    tags = Table('tags', metadata,
        Column('id', String, primary_key=True),
        Column('name', String, unique=True, nullable=False)
    )

    question_tags = Table('question_tags', metadata,
        Column('question_id', String, ForeignKey('questions.id'), primary_key=True),
        Column('tag_id', String, ForeignKey('tags.id'), primary_key=True)
    )

    # Create the tables in the database
    try:
        with engine.connect() as connection:
            metadata.create_all(engine)
            print("Tables created successfully (if they didn't exist).")
    except Exception as e:
        print(f"Error creating tables: {e}")
        return


    # Load the enriched data
    try:
        df = pd.read_csv(ENRICHED_CSV)
    except FileNotFoundError:
        print(f"Error: {ENRICHED_CSV} not found.")
        return

    # --- Data insertion ---
    with engine.connect() as connection:
        # Insert categories
        if 'category' in df.columns:
            unique_categories = df['category'].dropna().unique()
            for cat_name in unique_categories:
                # Check if category already exists
                existing = connection.execute(categories.select().where(categories.c.name == cat_name)).first()
                if not existing:
                    connection.execute(categories.insert().values(id=cat_name.lower().replace(' ', '_'), name=cat_name))
            print(f"Inserted {len(unique_categories)} categories.")

        # Insert tags
        if 'tags' in df.columns:
            all_tags = set()
            df['tags'].dropna().apply(lambda x: all_tags.update(eval(x) if isinstance(x, str) and x.startswith('[') else x.split(',')))
            for tag_name in all_tags:
                tag_name = tag_name.strip()
                # Check if tag already exists
                existing = connection.execute(tags.select().where(tags.c.name == tag_name)).first()
                if not existing:
                    connection.execute(tags.insert().values(id=tag_name.lower().replace(' ', '_'), name=tag_name))
            print(f"Inserted {len(all_tags)} tags.")

        # Insert questions and question_tags
        for _, row in df.iterrows():
            # Prepare question data
            question_data = {
                'id': row['id'],
                'title': row['question'].split('\n')[0],
                'body': '\n'.join(row['question'].split('\n')[1:]),
                'difficulty': row['difficulty'].upper() if isinstance(row['difficulty'], str) else 'EASY', # Default to easy if null
                'category_id': str(row['category']).lower().replace(' ', '_') if pd.notna(row['category']) else None,
                'tags_text': str(row['tags']) if pd.notna(row['tags']) else None
            }
            # Check if question already exists
            existing_question = connection.execute(questions.select().where(questions.c.id == question_data['id'])).first()
            if not existing_question:
                connection.execute(questions.insert().values(question_data))

            # Prepare and insert tags for the question
            if pd.notna(row['tags']):
                try:
                    q_tags = eval(row['tags']) if isinstance(row['tags'], str) and row['tags'].startswith('[') else row['tags'].split(',')
                    for tag_name in q_tags:
                        tag_name = tag_name.strip()
                        tag_id = tag_name.lower().replace(' ', '_')
                        # Check if question_tag link already exists
                        existing_link = connection.execute(question_tags.select().where(question_tags.c.question_id == row['id']).where(question_tags.c.tag_id == tag_id)).first()
                        if not existing_link:
                            connection.execute(question_tags.insert().values(question_id=row['id'], tag_id=tag_id))
                except Exception as e:
                    print(f"Could not parse tags for question {row['id']}: {row['tags']}. Error: {e}")

        print("Data migration complete.")


if __name__ == "__main__":
    main()
