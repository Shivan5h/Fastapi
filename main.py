from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI  # Updated import
from langchain.schema import HumanMessage

# Load the user data from CSV
df = pd.read_csv('user.csv')  # Adjust the path to your CSV file

# FastAPI app
app = FastAPI()

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Pydantic model to validate input data
class UserInput(BaseModel):
    user_id: int
    location: str
    favourite_cuisine: str
    veg_or_nonveg: str
    no_of_followers: int  # Add no_of_followers to the input model

# Function to calculate distance between two cities using GPT
openai_api_key = "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"

def gpt_city_distance(city1, city2, openai_api_key):
    try:
        # Create a prompt template for asking the distance
        prompt_template = PromptTemplate(
            input_variables=["city1", "city2"],
            template="Estimate the distance in kilometers between {city1} and {city2} in India."
        )

        # Format the prompt with the two cities
        prompt = prompt_template.format(city1=city1, city2=city2)

        # Initialize the ChatOpenAI model with the API key
        chat_model = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0,
            max_retries=3
        )

        # Send the prompt to GPT and get the response
        response = chat_model([HumanMessage(content=prompt)])

        # Extract the distance from GPT's response content
        distance_str = response.content.strip()

        # Convert the response into a float (assuming the distance is the first word)
        try:
            distance = float(distance_str.split()[0])
            return distance
        except ValueError:
            print(f"Unable to parse distance from response: '{distance_str}'. Assigning default value 1000.")
            return 1000  # Default large distance if parsing fails

    except Exception as e:
        print(f"Error in GPT city distance calculation: {e}")
        return 1000  # Default large distance in case of any error

# Function to calculate similarity score
def calculate_similarity(user_input: UserInput, user_row: pd.Series, openai_api_key: str):
    similarity_score = 0

    # Cuisine similarity (High priority)
    if user_input.favourite_cuisine.lower() == user_row['favourite_cuisine'].lower():
        similarity_score += 50

    # Veg/Non-Veg preference match (Medium priority)
    if user_input.veg_or_nonveg.lower() == user_row['veg_or_nonveg'].lower():
        similarity_score += 30

    # Location similarity using GPT for estimating distance
    location_distance = gpt_city_distance(user_input.location, user_row['location'], openai_api_key)
    location_similarity = 1 / (1 + location_distance)  # Inverse relationship with distance
    similarity_score += location_similarity * 10  # Small weight for location

    # Followers count influence
    try:
        followers_weight = int(user_input.no_of_followers) / 5000
    except (ValueError, TypeError):
        followers_weight = 0  # Handle invalid followers count gracefully
    similarity_score += followers_weight * 10  # Small weight for followers

    return similarity_score

# Endpoint to calculate and return recommendations
@app.post("/recommend_users")
async def recommend_users(input_user: UserInput, openai_api_key: str):
    if not openai_api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key is required.")

    similarity_scores = []

    # Iterate over each user in the dataset
    for _, user_row in df.iterrows():
        if input_user.user_id == user_row['user_id']:  # Skip the same user
            continue

        # Calculate similarity score
        similarity_score = calculate_similarity(input_user, user_row, openai_api_key)
        similarity_scores.append((user_row['user_id'], similarity_score))

    # Sort by similarity score in descending order
    sorted_similarities = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Return top 10 recommendations
    return {
        "input_user": input_user.user_id,
        "recommended_users": [{"user_id": user[0], "similarity_score": user[1]} for user in sorted_similarities[:10]]
    }

# To run the app, use: uvicorn main:app --reload
