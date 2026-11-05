import os
import functools
import google.generativeai as genai

from dotenv import load_dotenv
from loguru import logger
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.llms.mock import MockLLM
from llama_index.core.embeddings import MockEmbedding

# load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

def is_valid_key(key: str) -> bool:
    if not key:
        return False
    key_clean = key.strip().lower()
    return "your_" not in key_clean and key_clean != ""

if is_valid_key(api_key):
    genai.configure(api_key=api_key)

logger.add("logs/app.log", rotation="10 MB")

# Configure LLM and EmbedModel
try:
    if is_valid_key(api_key):
        Settings.llm = Gemini(model="models/gemini-3.5-flash", api_key=api_key)
        Settings.embed_model = GeminiEmbedding(model_name="models/gemini-embedding-2", api_key=api_key)
    else:
        logger.warning("No valid Gemini API key found. Using Mock models.")
        Settings.llm = MockLLM()
        Settings.embed_model = MockEmbedding(embed_dim=1536)
except Exception as e:
    logger.error(f"Error initializing LlamaIndex settings: {str(e)}")
    Settings.llm = MockLLM()
    Settings.embed_model = MockEmbedding(embed_dim=1536)


@functools.lru_cache(maxsize=1)
def build_food_index(data_path: str = "data") -> VectorStoreIndex:
    """Build the vector index once and cache it for the lifetime of the process."""
    try:
        documents = SimpleDirectoryReader(data_path).load_data()

        index = VectorStoreIndex.from_documents(
            documents
        )

        logger.info("Food Knowledge index built successfully (cached)")

        return index

    except Exception as e:
        logger.error(f"Error building index: {str(e)}")
        # If it's a quota error, immediately switch to MockEmbedding to build index offline/demo!
        if "quota" in str(e).lower() or "limit" in str(e).lower() or "429" in str(e):
            logger.info("Switching to MockEmbedding due to quota error")
            try:
                from llama_index.core.embeddings import MockEmbedding
                Settings.embed_model = MockEmbedding(embed_dim=1536)
                index = VectorStoreIndex.from_documents(documents)
                return index
            except Exception as e2:
                logger.error(f"Failed to build even with MockEmbedding: {str(e2)}")
        return None


def query_nutrition_knowledge(question: str) -> str:
    if not is_valid_key(api_key):
        return get_mock_knowledge_answer(question)
    try:
        index = build_food_index()  # returns cached index after first call

        if not index:
            # Fallback to mock on build failure
            return get_mock_knowledge_answer(question)

        query_engine = index.as_query_engine()

        response = query_engine.query(question)

        logger.info(f"Query executed: {question}")

        return str(response)

    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
        if "quota" in str(e).lower() or "limit" in str(e).lower() or "429" in str(e) or "api_key" in str(e).lower() or "key not valid" in str(e).lower():
            return get_mock_knowledge_answer(question)
        return "Unable to answer the question at the moment"


def get_nutrition_info(food_item: str) -> str:
    if not is_valid_key(api_key):
        return get_mock_nutrition_data(food_item)
    try:
        model = genai.GenerativeModel(
            model_name="gemini-3.5-flash",
            system_instruction="you are a food nutrition expert"
        )
        prompt = f"""Give me detailed nutrition info for: {food_item}

Return the information in exactly this structure using markdown tables:

## {food_item} — Nutrition Facts

**Overview**: [Write 1-2 sentences describing this food's nutritional character]

### Macronutrients (per 100g serving)
| Nutrient | Amount | % Daily Value |
|---|---|---|
| Calories | ... kcal | ... % |
| Protein | ... g | ... % |
| Total Carbohydrates | ... g | ... % |
| — Dietary Fiber | ... g | ... % |
| — Sugars | ... g | — |
| Total Fat | ... g | ... % |
| — Saturated Fat | ... g | ... % |
| — Unsaturated Fat | ... g | — |
| Water | ... g | — |

### Key Micronutrients
| Nutrient | Amount | % Daily Value |
|---|---|---|
[List the most relevant 6-8 vitamins and minerals for this food]

### Health Highlights
[3 concise bullet points on key health benefits or considerations]
"""
        response = model.generate_content(prompt)
        nutrition_data = response.text

        logger.info(
            f"Fetched nutrition data for {food_item}"
        )

        return nutrition_data

    except Exception as e:
        logger.error(f"Gemini error: {str(e)}")
        if "quota" in str(e).lower() or "limit" in str(e).lower() or "429" in str(e) or "api_key" in str(e).lower() or "key not valid" in str(e).lower():
            return get_mock_nutrition_data(food_item)
        return "Unable to fetch nutrition info"


def get_mock_nutrition_data(food_item: str) -> str:
    food_clean = food_item.strip().lower()
    
    mocks = {
        "banana": """### 🍌 Nutrition Analysis for Banana (Medium, ~118g)
 
**⚠️ [Demo Mode: Gemini Quota Exceeded]** *This is a high-fidelity simulated response.*
 
*   **Calories**: 105 kcal
*   **Macronutrients**:
    *   **Carbohydrates**: 27g (with 3g Fiber, 14g Sugars)
    *   **Protein**: 1.3g
    *   **Fat**: 0.3g (Negligible)
*   **Micronutrients & Benefits**:
    *   **Potassium**: 422mg (9% DV) — crucial for blood pressure regulation and heart function.
    *   **Vitamin B6**: 33% DV — supports metabolism and cognitive development.
    *   **Vitamin C**: 11% DV — helps support immune function.
*   **Expert Verdict**: Bananas are an excellent pre-workout snack, providing easily digestible carbohydrates and potassium to prevent muscle cramps.""",
        "apple": """### 🍎 Nutrition Analysis for Apple (Medium, ~182g)
 
**⚠️ [Demo Mode: Gemini Quota Exceeded]** *This is a high-fidelity simulated response.*
 
*   **Calories**: 95 kcal
*   **Macronutrients**:
    *   **Carbohydrates**: 25g (with 4.4g Fiber, 19g Sugars)
    *   **Protein**: 0.5g
    *   **Fat**: 0.3g
*   **Micronutrients & Benefits**:
    *   **Vitamin C**: 14% DV — powerful antioxidant support.
    *   **Pectin**: High soluble fiber content that supports gut health and lowers cholesterol.
*   **Expert Verdict**: Apples are low in glycemic index and high in water and fiber, making them excellent for satiety and cardiovascular health.""",
        "avocado toast": """### 🥑 Nutrition Analysis for Avocado Toast (1 slice whole wheat bread + 1/2 avocado)
 
**⚠️ [Demo Mode: Gemini Quota Exceeded]** *This is a high-fidelity simulated response.*
 
*   **Calories**: 290 kcal
*   **Macronutrients**:
    *   **Carbohydrates**: 22g (with 8g Fiber)
    *   **Protein**: 6g
    *   **Fat**: 16g (mostly heart-healthy monounsaturated fats)
*   **Micronutrients & Benefits**:
    *   **Healthy Fats**: Oleic acid reduces inflammation and supports cardiovascular wellness.
    *   **Folate & Vitamin K**: Essential for blood clotting and cell health.
*   **Expert Verdict**: Avocado toast is a balanced breakfast option, providing sustained energy through fiber, healthy fats, and complex carbohydrates."""
    }
    
    default_response = f"""### 🍽️ Nutrition Analysis for {food_item.title()}
 
**⚠️ [Demo Mode: Gemini Quota Exceeded]** *This is a high-fidelity simulated response.*
 
*   **Calories**: ~180-250 kcal (estimated)
*   **Macronutrients**:
    *   **Carbohydrates**: Moderate
    *   **Protein**: Moderate
    *   **Fat**: Low to Moderate (depends on preparation)
*   **Key Health Highlights**:
    *   Provides active micronutrients, vitamins, and energy.
    *   High in general nutritional value depending on culinary preparation.
*   **Expert Verdict**: To optimize the nutrition of `{food_item}`, focus on lean preparation methods (steaming, grilling, baking) and pair with fiber-rich vegetables."""
    
    return mocks.get(food_clean, default_response)
 
 
def get_mock_knowledge_answer(question: str) -> str:
    q_clean = question.strip().lower()
    
    if "calorie" in q_clean or "intake" in q_clean:
        return """**⚠️ [Demo Mode: Gemini Quota Exceeded]** *Simulated answer based on your uploaded documentation (data/nutrition_basics.txt):*
 
According to the local database, the general reference daily caloric intake for an average adult is **2,000 calories per day**. This needs to be adjusted based on individual factors such as age, gender, activity levels, and personal health goals. The target macronutrient distribution recommendation is:
- **Carbohydrates**: 45–65% of daily calories
- **Proteins**: 10–35% of daily calories
- **Fats**: 20–35% of daily calories"""
    elif "avocado" in q_clean or "superfood" in q_clean:
        return """**⚠️ [Demo Mode: Gemini Quota Exceeded]** *Simulated answer based on your uploaded documentation (data/nutrition_basics.txt):*
 
According to the local database, **Avocados** are listed as a superfood. They are highly beneficial because they are:
- Rich in **monounsaturated fats** (specifically heart-healthy oleic acid, which helps reduce inflammation and supports heart health).
- High in **fiber** (supporting healthy digestion).
- High in **potassium** (essential for blood pressure regulation)."""
    else:
        return """**⚠️ [Demo Mode: Gemini Quota Exceeded]** *Simulated answer based on local knowledge:*
 
Based on general nutrition guidelines:
- Maintain a balanced diet rich in whole foods, lean proteins, healthy fats, and fiber.
- Keep sodium intake under 2,300 mg daily for heart health.
- Drink 2 to 2.5 liters of water daily for optimal hydration.
 
*For precise answers, configure an active Gemini API key with credit balance in your `.env` file.*"""
